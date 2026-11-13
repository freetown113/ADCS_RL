import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .cli import observations_from_file


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('inputs',nargs='+',help='observation data from different regimes/configs')
    p.add_argument('--out',required=True)
    p.add_argument('--per-source',type=int,default=50000)
    p.add_argument('--seed',type=int,default=2001)
    args=p.parse_args(argv)
    
    if args.per_source <1:
        raise ValueError('--per-source must be positive')
    
    rng=np.random.default_rng(args.seed)
    arrays=[]
    labels=[]
    manifest=[]
    columns=None
    contract=None
    
    for idx,name in enumerate(args.inputs):
        obs=np.asarray(observations_from_file(name),np.float32)
        dataset_contract=None

        if Path(name).suffix=='.npz':
            with np.load(name,allow_pickle=False) as z:
                if 'observation_contract' in z:
                    dataset_contract=str(z['observation_contract'].item())

        if contract is None and idx==0:
            contract=dataset_contract
        elif dataset_contract!=contract:
            raise ValueError('incompatible/missing observation_contract between datasets')
        if obs.ndim!=2 or not obs.shape[0] or not np.isfinite(obs).all():
            raise ValueError(f'bad observation data in {name}')
        if columns is None:
            columns=obs.shape[1]
        if obs.shape[1]!=columns:
            raise ValueError('mixed actor input dimensions')
        chosen=np.sort(rng.choice(len(obs),size=min(len(obs),args.per_source),replace=False))
        arrays.append(obs[chosen])
        labels.append(np.full(len(chosen),idx,np.int32))
        manifest.append({'name':name,'rows_total':len(obs),'rows_used':len(chosen),
                         'sha256':hashlib.sha256(Path(name).read_bytes()).hexdigest()})
        
    obs=np.concatenate(arrays)
    ids=np.concatenate(labels)
    order=rng.permutation(len(obs))
    obs=obs[order]
    ids=ids[order]
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if contract is not None:
        np.savez_compressed(out,observations=obs,source_index=ids,
                            observation_contract=np.asarray(contract))
    else:
        np.savez_compressed(out,observations=obs,source_index=ids)

    out.with_suffix('.json').write_text(json.dumps({'sources':manifest,'rows':len(obs),'dimensions':columns,
                                                     'random_seed':args.seed,'observation_contract':json.loads(contract) if contract else None,
                                                     'note':'Distinct held-out scenario/seed trajectories are required for evaluation.'},indent=2))
    print(f'{len(obs)} balanced observations ({columns} floats) -> {out}')


if __name__=='__main__':
    main()
