import argparse
import json
import pickle
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .core import extract_haiku_actor, predict
from .binary import load_binary


def _env_and_ref(checkpoint, project):
    import importlib
    import jax
    pkg=importlib.import_module(project)
    cfg=importlib.import_module(project+'.config').config_from_dict(checkpoint['config'])
    env=importlib.import_module(project+'.env').SatelliteEnv(cfg)
    net=importlib.import_module(project+'.network').make_network(env.observation_size,env.action_size,cfg.network)
    return env,net,cfg


def _load(path):
    with open(path,'rb') as f: return pickle.load(f)


def _record(env,params,network,seed,episodes,max_samples):
    import jax
    import jax.numpy as jnp
    from importlib import import_module

    control=import_module(env.__class__.__module__.rsplit('.',1)[0]+'.control')
    state=env.reset(jax.random.PRNGKey(seed),episodes)
    def step_fn(state,_):
        obs=env.observe(state)
        mean,_=network.apply(params,obs)
        action=jnp.tanh(mean)
        command=control.policy_to_command(env,state,action)
        state,_,_,_=env.step(state,command)
        return state,obs
    _,observations=jax.jit(lambda s: jax.lax.scan(step_fn,s,None,length=env.episode_steps))(state)
    obs=np.asarray(jax.device_get(observations),np.float32).reshape(-1,env.observation_size)

    if max_samples is not None:
        idx=np.linspace(0,len(obs)-1,min(max_samples,len(obs)),dtype=np.int64)
        obs=obs[idx]
    return obs


def _closed_loop(env,model):
    import jax
    import jax.numpy as jnp

    layers=tuple((jnp.asarray(d.weight),jnp.asarray(d.bias),
                  None if d.weight_scales is None else jnp.asarray(d.weight_scales),
                  None if d.input_scale is None else jnp.asarray(d.input_scale))
                 for d in model.layers)
    def apply(p,obs):
        x=jnp.asarray(obs,dtype=jnp.float32)
        for i,(w,b,ws,xs) in enumerate(p):
            if model.variant=='w8a8':
                qx=jnp.clip(jnp.rint(x/xs),-127,127).astype(jnp.int32)
                y=(qx@w.astype(jnp.int32)+b.astype(jnp.int32)).astype(jnp.float32)*(xs*ws)
            elif model.variant.startswith('w8'):
                weff=w.astype(jnp.float32)*ws[None,:]
                if model.variant=='w8a16':
                    x=x.astype(jnp.float16).astype(jnp.float32)
                    weff=weff.astype(jnp.float16).astype(jnp.float32)
                    y=x@weff+b.astype(jnp.float32)
                    y=y.astype(jnp.float16).astype(jnp.float32)
                else:
                    y=x@weff+b.astype(jnp.float32)
            else:
                if model.variant=='fp16':
                    x=x.astype(jnp.float16).astype(jnp.float32)
                    y=x@w.astype(jnp.float32)+b.astype(jnp.float32)
                    y=y.astype(jnp.float16).astype(jnp.float32)
                elif model.variant=='bf16':
                    x=x.astype(jnp.bfloat16).astype(jnp.float32)
                    # Decode BF16 words to f32, accumulate in FP32.
                    ww=jax.lax.bitcast_convert_type(jnp.asarray(w,dtype=jnp.uint16).astype(jnp.uint32) << jnp.uint32(16),jnp.float32)
                    bb=jax.lax.bitcast_convert_type(jnp.asarray(b,dtype=jnp.uint16).astype(jnp.uint32) << jnp.uint32(16),jnp.float32)
                    y=x@ww+bb
                    y=y.astype(jnp.bfloat16).astype(jnp.float32)
                else:
                    y=x@w+b
            if i<len(p)-1:
                x=jnp.tanh(y) if model.activation=='tanh' else jnp.maximum(y,0)
                if model.variant in ('fp16','w8a16'):x=x.astype(jnp.float16).astype(jnp.float32)
                elif model.variant=='bf16':x=x.astype(jnp.bfloat16).astype(jnp.float32)
            else:x=y
        mean=model.mean_limit*jnp.tanh(x/model.mean_limit)
        return mean,jnp.zeros(mean.shape[:-1],dtype=mean.dtype)
    return layers,apply


def main(argv=None):
    parser=argparse.ArgumentParser()
    subs=parser.add_subparsers(dest='cmd',required=True)
    for cmd in ('collect','evaluate'):
        p=subs.add_parser(cmd)
        p.add_argument('checkpoint')
        p.add_argument('--project',default='adcs_flightlike_b1')
        p.add_argument('--episodes',type=int,default=64)
        p.add_argument('--seed',type=int,default=None)
        if cmd=='collect':
            p.add_argument('--out',required=True)
            p.add_argument('--max-samples',type=int,default=200000)
        else:
            p.add_argument('--models',required=True)
            p.add_argument('--out',default='closed_loop.json')
    args=parser.parse_args(argv)

    import jax
    ckpt=_load(args.checkpoint)
    
    env,network,cfg=_env_and_ref(ckpt,args.project)
    seed=args.seed if args.seed is not None else cfg.run.seed+(20000 if args.cmd=='collect' else 10000)
    actor=extract_haiku_actor(ckpt['params'],hidden_count=len(cfg.network.hidden_sizes),
                              activation=cfg.network.activation,mean_limit=cfg.network.policy_mean_limit)
    if actor.input_size!=env.observation_size or actor.output_size!=env.action_size:
        raise ValueError('checkpoint policy dimensions incompatible with environment')
    
    if args.cmd=='collect':
        x=_record(env,ckpt['params'],network,seed,args.episodes,args.max_samples)
        out=Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        contract = {'schema': 'B4_env.observe_v1',
                    'observation_size': env.observation_size,
                    'action_size': env.action_size,
                    'control_mode': cfg.control.mode,
                    'observation_config': asdict(cfg.observation),
                    'omega_scale': cfg.observation.omega_scale,
                    'max_wheel_speed': cfg.physics.max_wheel_speed}
        np.savez_compressed(out, observations=x,
                            observation_contract=np.asarray(json.dumps(contract,sort_keys=True)))
        original,_=network.apply(ckpt['params'],jax.numpy.asarray(x[:min(512,len(x))]))
        delta=np.max(np.abs(np.asarray(jax.numpy.tanh(original))-predict(actor,x[:min(512,len(x))])))
        print(f'Rows {len(x)}, cols {x.shape[1]}, actor/ref action max error {delta:.3g}: {out}')
        if delta>5e-5:
            raise RuntimeError('actor extraction failed reference check')
    else:
        from importlib import import_module
        evaluation=import_module(args.project+'.evaluation')
        state=env.reset(jax.random.PRNGKey(seed),args.episodes)
        rows=[]
        _,base_traj,base_summary=jax.jit(lambda s:evaluation.evaluate_policy(env,ckpt['params'],network.apply,s))(state)
        def summarize(metrics):
            return {k:float(v) for k,v in jax.device_get(metrics).items()}
        rows.append({'variant':'fp32_haiku_reference','binary_bytes':None,**summarize(base_summary)})
        for f in sorted(Path(args.models).glob('*/actor.bin')):
            model=load_binary(f)
            if model.input_size!=env.observation_size or model.output_size!=env.action_size:
                raise ValueError('incorrect actor shape')
            layers,apply=_closed_loop(env,model)
            _,traj,metrics=jax.jit(lambda p,s:evaluation.evaluate_policy(env,p,apply,s))(layers,state)
            row={'variant':model.variant,'binary_bytes':f.stat().st_size,**summarize(metrics)}
            angle=np.asarray(jax.device_get(traj.angle_rad))
            row['rms_pointing_error_deg']=float(np.rad2deg(np.sqrt(np.mean(angle**2))))
            rows.append(row)
            print(json.dumps(row))
        output_path=Path(args.out)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(json.dumps(rows,indent=2))
        print(args.out)
        import csv
        with output_path.with_suffix('.csv').open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)


if __name__=='__main__':
    main()
