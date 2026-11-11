import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from .binary import load_binary
from .core import predict
from .cli import observations_from_file



def compare(binary,observations,rust_binary,atol,chunk=1000):
    m=load_binary(binary)
    x=np.asarray(observations,np.float32)
    if x.ndim==1:
        x = x[None,:]
    reference=predict(m,x)
    proc=subprocess.Popen([str(rust_binary),'--model',str(binary),'--stdio'],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,bufsize=1)
    diffs=[]
    try:
        for i,row in enumerate(x):
            req=json.dumps({'id':i,'observations':row.astype(float).tolist()})+'\n'
            proc.stdin.write(req)
            proc.stdin.flush()
            output=proc.stdout.readline()
            if not output:
                raise RuntimeError('Rust service exited unexpectedly: '+proc.stderr.read())
            response=json.loads(output)
            if response.get('id')!=i or 'error' in response:
                raise RuntimeError(f'bad Rust response: {response}')
            action=np.asarray(response['action'],np.float32)
            if action.shape!=(m.output_size,):
                raise AssertionError('output shape mismatch')
            diffs.append(np.abs(action-reference[i]))
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    diff=np.asarray(diffs)
    result={'model':str(binary),'variant':m.variant,'samples':len(x),
            'max_abs':float(diff.max()),'mae':float(diff.mean()),
            'tolerance':atol,'pass':bool(diff.max()<=atol)}
    
    if not result['pass']:
        raise AssertionError(f'Rust / NumPy equivalence FAIL: {result}')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('observations',help='held-out .npy/.npz observations')
    p.add_argument('--models',required=True)
    p.add_argument('--rust-binary',required=True)
    p.add_argument('--max-samples',type=int,default=10000)
    p.add_argument('--atol',type=float,default=3.e-4)
    p.add_argument('--out',default='rust_equivalence.json')
    args=p.parse_args()
    x=observations_from_file(args.observations)
    
    if x.ndim==1:
        x=x[None,:]
    if args.max_samples<len(x):
        rng=np.random.default_rng(1107)
        x=x[np.sort(rng.choice(len(x),args.max_samples,replace=False))]
    results=[]

    for f in sorted(Path(args.models).glob('*/actor.bin')):
        row=compare(f,x,Path(args.rust_binary),args.atol)
        results.append(row)
        print(json.dumps(row))
    Path(args.out).write_text(json.dumps(results,indent=2))



if __name__=='__main__':
    main()
