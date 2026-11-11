import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from .core import Actor, Dense, extract_haiku_actor, predict, calibrate, quantize, VARIANTS
from .binary import save_binary, load_binary, write_metadata


def observations_from_file(path):
    path = Path(path)
    if path.suffix == '.npy': 
        return np.load(path, allow_pickle=False)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            if 'observations' not in z: raise ValueError('npz must have observations key')
            return z['observations']
    if path.suffix == '.csv': 
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    raise ValueError('observations must be .npy, .npz or numeric .csv')


def load_actor(path):
    p = Path(path)
    if p.suffix == '.npz':
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            layers = [Dense(z[f'w{i}'].astype(np.float32), z[f'b{i}'].astype(np.float32))
                      for i in range(meta['layers'])]
            a = Actor(layers, meta['activation'], meta['policy_mean_limit'])
    elif p.suffix == '.pkl':
        with p.open('rb') as f: checkpoint = pickle.load(f)
        if 'params' not in checkpoint: 
            raise ValueError('checkpoint missing params')
        cfg = checkpoint['config']['network']
        a = extract_haiku_actor(checkpoint['params'], hidden_count=len(cfg['hidden_sizes']),
                                activation=cfg['activation'], mean_limit=cfg['policy_mean_limit'])
    else: 
        raise ValueError('actor file must be trusted B4 .pkl or standalone .npz')
    a.validate()
    return a


def save_actor(actor, path):
    metadata = {'layers': len(actor.layers), 'activation': actor.activation,
                'policy_mean_limit': actor.mean_limit}
    args = {f'w{i}': d.weight for i,d in enumerate(actor.layers)}
    args.update({f'b{i}': d.bias for i,d in enumerate(actor.layers)})
    np.savez(path, metadata=np.asarray(json.dumps(metadata)), **args)


def _metrics(reference, other):
    delta = np.asarray(other, np.float32) - np.asarray(reference, np.float32)
    return {'action_mae': float(np.abs(delta).mean()), 'action_rmse': float(np.sqrt((delta*delta).mean())),
            'action_max_abs': float(np.abs(delta).max())}


def main(argv=None):
    p=argparse.ArgumentParser(prog='python -m adcs_embed', description='Standalone ADCS actor quantization/export')
    sub=p.add_subparsers(dest='command', required=True)
    x=sub.add_parser('extract',help='Extract actor-only .npz from a trusted Haiku checkpoint')
    x.add_argument('checkpoint')
    x.add_argument('--out',required=True)
    x=sub.add_parser('calibrate',help='Calibrate from recorded FP32-closed-loop observations')
    x.add_argument('actor')
    x.add_argument('observations')
    x.add_argument('--out', required=True)
    x.add_argument('--percentile',type=float,default=100.)
    x.add_argument('--margin',type=float,default=1.05)
    x=sub.add_parser('export',help='Export all or selected precision variants as binary models')
    x.add_argument('actor')
    x.add_argument('--calibration',required=True)
    x.add_argument('--out',required=True)
    x.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    x=sub.add_parser('benchmark',help='Held-out open-loop tests; not a closed-loop success benchmark')
    x.add_argument('actor')
    x.add_argument('observations')
    x.add_argument('--models',required=True)
    x.add_argument('--out',required=True)
    x.add_argument('--repeats',type=int,default=1000)
    x=sub.add_parser('vectors',help='Export JSONL reference vectors for Rust cross-language tests')
    x.add_argument('observations')
    x.add_argument('--model',required=True)
    x.add_argument('--out',required=True)

    args=p.parse_args(argv)
    if args.command=='extract':
        a=load_actor(args.checkpoint)
        save_actor(a,args.out)
        print(f'Actor-only: {a.input_size} -> {[d.weight.shape[1] for d in a.layers]}, {args.out}')
    elif args.command=='calibrate':
        a=load_actor(args.actor)
        obs=observations_from_file(args.observations)
        cal=calibrate(a,obs,percentile=args.percentile,margin=args.margin)

        if Path(args.observations).suffix=='.npz':
            with np.load(args.observations,allow_pickle=False) as z:
                if 'observation_contract' in z:
                    contract=json.loads(str(z['observation_contract'].item()))
                    if (contract.get('observation_size')!=a.input_size or
                        contract.get('action_size')!=a.output_size):
                        raise ValueError('calibration dataset observation/action contract does not match actor')
                    cal['observation_contract']=contract

        cal['observation_file_sha256']=hashlib.sha256(Path(args.observations).read_bytes()).hexdigest()
        Path(args.out).write_text(json.dumps(cal,indent=2))
        print(json.dumps(cal,indent=2))
    elif args.command=='export':
        a=load_actor(args.actor)
        c=json.loads(Path(args.calibration).read_text())
        root=Path(args.out)
        contract=c.get('observation_contract')
        if contract is not None and (contract.get('observation_size')!=a.input_size or
                                     contract.get('action_size')!=a.output_size):
            raise ValueError('calibration contract incompatible with actor')
        digest=hashlib.sha256()
        digest.update(json.dumps({'activation':a.activation,'mean_limit':a.mean_limit},sort_keys=True).encode())
        for d in a.layers:
            digest.update(np.ascontiguousarray(d.weight,dtype='<f4').tobytes())
            digest.update(np.ascontiguousarray(d.bias,dtype='<f4').tobytes())
        actor_sha256=digest.hexdigest()

        for variant in args.variants:
            m=quantize(a,variant,c)
            directory=root/variant
            directory.mkdir(parents=True,exist_ok=True)
            size=save_binary(m,directory/'actor.bin')
            write_metadata(directory/'model.json',m,calibration=c if variant=='w8a8' else None,
                           origin={'source':str(args.actor),'actor_sha256':actor_sha256,
                                   'calibration_sha256':hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()})

            check=load_binary(directory/'actor.bin')
            probe=np.zeros((2,a.input_size),np.float32)
            assert np.array_equal(predict(m,probe),predict(check,probe))
            print(f'{variant:6s} {size:8d} bytes -> {directory}')
    elif args.command=='benchmark':
        a=load_actor(args.actor)
        obs=observations_from_file(args.observations)

        if obs.ndim==1: 
            obs=obs[None,:]
        if not len(obs): 
            raise ValueError('empty held-out observations')
        ref=predict(a,obs)
        rows=[]
        
        for variant in VARIANTS:
            f=Path(args.models)/variant/'actor.bin'
            if not f.exists():
                continue
            model=load_binary(f)
            start=time.perf_counter_ns()
            result=predict(model,obs)
            batch_us=(time.perf_counter_ns()-start)/1000/len(obs)
            single=obs[:1]
            n=max(1,args.repeats)
            times=[]

            for _ in range(n):
                t=time.perf_counter_ns()
                predict(model,single)
                times.append((time.perf_counter_ns()-t)/1000)
            row=dict(variant=variant,binary_bytes=f.stat().st_size,
                     heldout_input_clip_fraction=(float(np.mean(np.abs(obs)>model.layers[0].input_scale*127)) if variant=='w8a8' else None),
                     numpy_batch_us_per_obs=batch_us,
                     numpy_single_p50_us=float(np.median(times)),
                     numpy_single_p99_us=float(np.percentile(times,99)),
                     numpy_single_max_us=float(max(times)),
                     **_metrics(ref,result))
            
            rows.append(row)
            print(json.dumps(row))
        out=Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.with_suffix('.json').write_text(json.dumps(rows,indent=2))
        with out.with_suffix('.csv').open('w',newline='') as h:
            wr=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else [])
            if rows: 
                wr.writeheader()
                wr.writerows(rows)

    elif args.command=='vectors':
        model=load_binary(args.model)
        obs=observations_from_file(args.observations)
        if obs.ndim==1:
            obs=obs[None,:]
        path=Path(args.out)
        path.parent.mkdir(parents=True,exist_ok=True)
        
        with path.open('w') as h:
            for i,row in enumerate(obs):
                action,mean=predict(model,row,return_mean=True)
                h.write(json.dumps({'id':i,'observations':row.astype(float).tolist(),
                                    'action':action.astype(float).tolist(),
                                    'mean':mean.astype(float).tolist()})+'\n')
        print(f'{len(obs)} reference vectors written: {path}')


if __name__=='__main__':
    main()
