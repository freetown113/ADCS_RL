import argparse
import json
import pickle
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .core import extract_haiku_actor, predict, float_hidden_outputs
from .binary import load_binary
from .sensor_metrics import trajectory_sensor_report, rms_pointing_error_deg


def _env_and_ref(checkpoint, project):
    import importlib
    import jax
    pkg=importlib.import_module(project)
    cfg=importlib.import_module(project+'.config').config_from_dict(checkpoint['config'])
    env=importlib.import_module(project+'.env').SatelliteEnv(cfg)
    net=importlib.import_module(project+'.network').make_network(env.observation_size,env.action_size,cfg.network)
    return env,net,cfg


def _load(path):
    with open(path,'rb') as f: 
        return pickle.load(f)

def _difference_statistics(actual, expected):
    """Return numeric differences without hiding exceptional observations."""
    delta = np.abs(np.asarray(actual, dtype=np.float32) -
                   np.asarray(expected, dtype=np.float32))
    if delta.shape != np.shape(expected) or not np.isfinite(delta).all():
        raise ValueError('incompatible or nonfinite reference actions')
    flat = int(np.argmax(delta))
    row, col = np.unravel_index(flat, delta.shape)
    return {'max_abs': float(delta[row, col]),
            'mean_abs': float(delta.mean()),
            'p99_abs': float(np.percentile(delta, 99)),
            'worst_batch_row': int(row),
            'worst_action_column': int(col)}


def _check_actor_against_haiku(network, params, actor, observations, *,
                               check_samples=4096, tolerance=5.e-5,
                               diagnostic_layers=True):
    """Compare original Haiku action, reimplemented JAX action, NumPy actor.

    Sampling across the *whole* observation file avoids checking only a few
    acquisition steps. This operates on a pre-recorded dataset, so no closed-
    loop rerun is necessary when investigating a mismatch.
    """
    import jax
    import jax.numpy as jnp
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != actor.input_size or not len(obs):
        raise ValueError('observation dimensions incompatible with extracted actor')
    if not np.isfinite(obs).all():
        raise ValueError('observation dataset contains NaN/Inf')
    if check_samples < 1 or not (np.isfinite(tolerance) and tolerance >= 0):
        raise ValueError('invalid check_samples or tolerance')
    positions = np.linspace(0, len(obs)-1, min(check_samples,len(obs)),dtype=np.int64)
    samples = obs[positions]
    jx = jnp.asarray(samples)

    original_mean, _ = network.apply(params, jx)
    original_mean = np.asarray(jax.device_get(original_mean), dtype=np.float32)
    original_action = np.asarray(jax.device_get(jnp.tanh(original_mean)), dtype=np.float32)
    numpy_action, numpy_mean = predict(actor, samples, return_mean=True)
    mean_difference = _difference_statistics(original_mean, numpy_mean)
    action_difference = _difference_statistics(original_action, numpy_action)

    # Recreate actor directly in JAX to distinguish topology/extraction issues
    # from backend-specific FP32 matmul arithmetic differences.
    y = jx
    jax_hidden = []
    for layer in actor.layers[:-1]:
        y = y @ jnp.asarray(layer.weight) + jnp.asarray(layer.bias)
        y = jnp.tanh(y) if actor.activation == 'tanh' else jnp.maximum(y, 0)
        if diagnostic_layers:
            jax_hidden.append(np.asarray(jax.device_get(y)))
    last = actor.layers[-1]
    raw = y @ jnp.asarray(last.weight) + jnp.asarray(last.bias)
    reconstructed_mean = actor.mean_limit * jnp.tanh(raw / actor.mean_limit)
    reconstructed_action = np.asarray(jax.device_get(jnp.tanh(reconstructed_mean)),dtype=np.float32)
    reconstruction_difference = _difference_statistics(original_action,reconstructed_action)

    result = {
        'status': 'PASS' if action_difference['max_abs'] <= tolerance else 'FAIL',
        'jax_backend': jax.default_backend(),
        'jax_default_matmul_precision': str(jax.config.jax_default_matmul_precision),
        'rows_in_file': int(len(obs)), 'rows_checked': int(len(samples)),
        'action_tolerance': float(tolerance),
        'haiku_vs_numpy_mean': mean_difference,
        'haiku_vs_numpy_action': action_difference,
        'haiku_vs_reconstructed_jax_action': reconstruction_difference,
        'worst_original_dataset_row': int(positions[action_difference['worst_batch_row']]),
    }
    if diagnostic_layers:
        numpy_hidden = float_hidden_outputs(actor, samples)
        result['hidden_layers_jax_vs_numpy'] = [
            {'layer': i, **_difference_statistics(j,h)}
            for i,(j,h) in enumerate(zip(jax_hidden,numpy_hidden))
        ]
    return result


def _check_report_or_raise(report, *, out=None):
    print(json.dumps(report, indent=2))
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f'Reference report: {path}')
    if report['status'] != 'PASS':
        raise RuntimeError(
            'Actor/Haiku reference check failed: '
            f"max action error {report['haiku_vs_numpy_action']['max_abs']:.6g} "
            f"> tolerance {report['action_tolerance']:.6g}. "
            'See layer diagnostics above. If using GPU, compare '
            '--matmul-precision highest versus default, do NOT raise the '
            'tolerance without explaining the discrepancy.'
        )

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
            if model.variant=='w8a8hi' and i == 0:
                y=x@(w.astype(jnp.float32)*ws[None,:])+b.astype(jnp.float32)
            elif model.variant in ('w8a8','w8a8hi'):
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
                if model.variant in ('fp16','w8a16'):
                    x=x.astype(jnp.float16).astype(jnp.float32)
                elif model.variant=='bf16':
                    x=x.astype(jnp.bfloat16).astype(jnp.float32)
            else:x=y
        mean=model.mean_limit*jnp.tanh(x/model.mean_limit)
        return mean,jnp.zeros(mean.shape[:-1],dtype=mean.dtype)
    return layers,apply


def main(argv=None):
    parser=argparse.ArgumentParser()
    subs=parser.add_subparsers(dest='cmd',required=True)
    for cmd in ('collect','evaluate','check'):
        p=subs.add_parser(cmd)
        p.add_argument('checkpoint')
        p.add_argument('--project',default='adcs_flightlike_b1')
        p.add_argument('--matmul-precision', choices=('highest','default','high'),
                        default='highest',
                        help='FP32 matrix multiplication mode for JAX, highest gives the closest CPU/Rust FP32 reference')
        p.add_argument('--check-samples',type=int,default=4096,
                        help='Equally spaced rows from the entire observation file for reference verification')
        p.add_argument('--action-tol',type=float,default=5e-5,
                        help='Maximum allowed action difference between original Haiku and NumPy FP32')
        if cmd != 'check':
            p.add_argument('--episodes',type=int,default=64)
            p.add_argument('--seed',type=int,default=None)
        if cmd=='collect':
            p.add_argument('--out',required=True)
            p.add_argument('--max-samples',type=int,default=200000)
            p.add_argument('--report',default=None, help='Save reference diagnostics as JSON')
        elif cmd=='check':
            p.add_argument('--observations',required=True,help='Previously collected .npz/.npy/.csv observations')
            p.add_argument('--report',default=None,help='Save reference diagnostics as JSON')
        else:
            p.add_argument('--models',required=True)
            p.add_argument('--out',default='closed_loop.json')
    args=parser.parse_args(argv)

    import jax
    jax.config.update('jax_default_matmul_precision', args.matmul_precision)
    ckpt=_load(args.checkpoint)
    
    env,network,cfg=_env_and_ref(ckpt,args.project)
    seed=(args.seed if args.seed is not None else cfg.run.seed +
          (20000 if args.cmd=='collect' else 10000)) if args.cmd != 'check' else None
    actor=extract_haiku_actor(ckpt['params'],hidden_count=len(cfg.network.hidden_sizes),
                              activation=cfg.network.activation,mean_limit=cfg.network.policy_mean_limit)
    if actor.input_size!=env.observation_size or actor.output_size!=env.action_size:
        raise ValueError('checkpoint policy dimensions incompatible with environment')
    
    if args.cmd=='check':
        from .cli import observations_from_file
        x=observations_from_file(args.observations)
        report=_check_actor_against_haiku(network,ckpt['params'],actor,x,
                                          check_samples=args.check_samples,
                                          tolerance=args.action_tol)
        _check_report_or_raise(report,out=args.report)
    elif args.cmd=='collect':
        x=_record(env,ckpt['params'],network,seed,args.episodes,args.max_samples)
        out=Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        contract = {'schema': 'env.observe_v1',
                    'observation_size': env.observation_size,
                    'action_size': env.action_size,
                    'control_mode': cfg.control.mode,
                    'observation_config': asdict(cfg.observation),
                    'omega_scale': cfg.observation.omega_scale,
                    'max_wheel_speed': cfg.physics.max_wheel_speed}
        np.savez_compressed(out, observations=x,
                            observation_contract=np.asarray(json.dumps(contract,sort_keys=True)))
        print(f'Collected {len(x)} observations × {x.shape[1]} features: {out}')
        report=_check_actor_against_haiku(network,ckpt['params'],actor,x,
                                          check_samples=args.check_samples,
                                          tolerance=args.action_tol)
        report['observations_path'] = str(out)
        _check_report_or_raise(report,out=args.report)
    else:
        from importlib import import_module
        evaluation=import_module(args.project+'.evaluation')
        state=env.reset(jax.random.PRNGKey(seed),args.episodes)
        rows=[]
        _,base_traj,base_summary=jax.jit(lambda s:evaluation.evaluate_policy(env,ckpt['params'],network.apply,s))(state)
        def summarize(metrics):
            numbers = {k: float(v) for k,v in jax.device_get(metrics).items()}
            return {k: v if np.isfinite(v) else None for k,v in numbers.items()}

        def rms_angle(trajectory):
            return rms_pointing_error_deg(jax.device_get(trajectory.angle_rad))

        def append_variant(label, binary_size, traj, summary):
            report = trajectory_sensor_report(traj,
                dt=float(env.config.physics.control_dt))
            row={'variant':label,'binary_bytes':binary_size,
                 **summarize(summary),
                 'rms_pointing_error_deg':rms_angle(traj)}
            # Flat, CSV-friendly supplementary age metrics. Detailed per-sensor
            # diagnostics are written to an additional JSON sidecar.
            for name, m in report.items():
                row[f'{name}_sample_age_max_finite_seconds']=m['finite_valid_age_max_seconds']
                row[f'{name}_nonfinite_valid_age_count']=m['nonfinite_age_while_valid_count']
                row[f'{name}_invalid_fraction']=1.-m['valid_fraction']
            rows.append(row)
            sensor_reports[label]=report
            print(json.dumps(row,allow_nan=False))

        sensor_reports={}
        append_variant('fp32_haiku_reference',None,base_traj,base_summary)        for f in sorted(Path(args.models).glob('*/actor.bin')):
            model=load_binary(f)
            if model.input_size!=env.observation_size or model.output_size!=env.action_size:
                raise ValueError('incorrect actor shape')
            layers,apply=_closed_loop(env,model)
            _,traj,metrics=jax.jit(lambda p,s:evaluation.evaluate_policy(env,p,apply,s))(layers,state)
            append_variant(model.variant,f.stat().st_size,traj,metrics)
        output_path=Path(args.out)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(json.dumps(rows,indent=2,allow_nan=False))
        print(args.out)
        age_path=output_path.with_name(output_path.stem+'_sensor_ages.json')
        age_path.write_text(json.dumps(sensor_reports,indent=2,allow_nan=False))
        print(f'Sensor availability and finite-age diagnostics: {age_path}')
        import csv
        with output_path.with_suffix('.csv').open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()