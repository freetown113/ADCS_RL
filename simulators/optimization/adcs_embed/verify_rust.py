import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from .binary import load_binary
from .core import predict
from .cli import observations_from_file
from .manifest import read_json, verify


def _bits(a):
    return np.ascontiguousarray(np.asarray(a,dtype=np.float32)).view(np.uint32).astype(np.uint64).tolist()


def numpy_trace(model,row):
    """Trace the NumPy reference with the same boundaries used by core.predict.
    This is a diagnostic reconstruction. The final action from core.predict is
    always used for the acceptance comparison, to avoid a second reference.
    """
    x=np.asarray(row,dtype=np.float32).reshape(1,-1)
    r={'preactivations':[],'preactivation_bits':[],'hidden_outputs':[],
       'hidden_output_bits':[],'quantized_inputs':[],'scaled_inputs':[],
       'scaled_input_bits':[],'integer_accumulators':[]}
    for li,layer in enumerate(model.layers):
        acc=None
        scaled=None
        qinput=None
        if model.variant=='w8a8hi' and li==0:
            weight=layer.weight.astype(np.float32)*layer.weight_scales[None,:]
            y=x@weight+layer.bias.astype(np.float32)
        elif model.variant in ('w8a8', 'w8a8hi'):
            scaled=x/np.float32(layer.input_scale)
            q=np.clip(np.rint(scaled),-127,127).astype(np.int8)
            acc=q.astype(np.int32)@layer.weight.astype(np.int32)+layer.bias.astype(np.int32)
            y=acc.astype(np.float32)*(np.float32(layer.input_scale)*layer.weight_scales)
            qinput=q[0].tolist()
        elif model.variant=='w8a32':
            weight=layer.weight.astype(np.float32)*layer.weight_scales[None,:]
            y=x@weight+layer.bias.astype(np.float32)
        else:
            raise ValueError('Detailed tracing supports w8a32/w8a8/w8a8hi')
        y=np.asarray(y,dtype=np.float32)
        r['preactivations'].append(y[0].astype(float).tolist())
        r['preactivation_bits'].append(_bits(y[0]))
        r['quantized_inputs'].append(qinput)
        r['scaled_inputs'].append(None if scaled is None else scaled[0].astype(float).tolist())
        r['scaled_input_bits'].append(None if scaled is None else _bits(scaled[0]))
        r['integer_accumulators'].append(None if acc is None else acc[0].tolist())
        if li+1<len(model.layers):
            x=(np.tanh(y) if model.activation=='tanh' else np.maximum(y,0)).astype(np.float32)
            r['hidden_outputs'].append(x[0].astype(float).tolist())
            r['hidden_output_bits'].append(_bits(x[0]))
        else:
            mean=np.float32(model.mean_limit)*np.tanh(y/np.float32(model.mean_limit))
            action=np.tanh(mean).astype(np.float32)
            r['mean']=mean[0].astype(float).tolist()
            r['action']=action[0].astype(float).tolist()
    return r


def _float_difference(a,b):
    if a is None or b is None:
        return None
    av=np.asarray(a,dtype=np.float32)
    bv=np.asarray(b,dtype=np.float32)
    if av.shape!=bv.shape:
        return {'shape_mismatch':[list(av.shape),list(bv.shape)]}
    delta=np.abs(av.astype(np.float64)-bv.astype(np.float64))
    i=int(np.argmax(delta))
    return {'max_abs':float(delta.flat[i]),'max_index':i,
            'reference':float(bv.flat[i]),'rust':float(av.flat[i]),
            'reference_bits':f'0x{int(bv.flat[i].view(np.uint32)):08x}',
            'rust_bits':f'0x{int(av.flat[i].view(np.uint32)):08x}'}


def _layer_comparison(reference,actual):
    layers=[]
    for li in range(len(reference['preactivations'])):
        layer={'layer':li,
            'preactivation':_float_difference(actual['preactivations'][li],reference['preactivations'][li]),
            'scaled_input':_float_difference(actual.get('scaled_inputs',[None]*len(reference['preactivations']))[li],
                                             reference['scaled_inputs'][li])}
        if li<len(reference['hidden_outputs']):
            layer['hidden_output']=_float_difference(actual['hidden_outputs'][li],reference['hidden_outputs'][li])
        qa=actual['quantized_inputs'][li]
        qb=reference['quantized_inputs'][li]
        aa=actual['integer_accumulators'][li]
        ab=reference['integer_accumulators'][li]

        if qa is not None and qb is not None:
            qa=np.asarray(qa)
            qb=np.asarray(qb)
            indices=np.flatnonzero(qa!=qb).tolist()
            layer['quant_code_differences']=len(indices)
            layer['differing_codes']=[]
            rs=reference['scaled_inputs'][li]
            ts=actual.get('scaled_inputs',[None]*len(reference['preactivations']))[li]

            for k in indices:
                item={'index':k,'numpy_code':int(qb[k]),'rust_code':int(qa[k])}
                if rs is not None and ts is not None:
                    rv=np.float32(rs[k])
                    tv=np.float32(ts[k])
                    item.update(numpy_scaled=float(rv),rust_scaled=float(tv),
                                numpy_bits=f'0x{rv.view(np.uint32):08x}',
                                rust_bits=f'0x{tv.view(np.uint32):08x}',
                                numpy_half_step_distance=float(abs(float(rv)-round(float(rv)-0.5)-0.5)))
                layer['differing_codes'].append(item)
        else:
            layer['quant_code_differences']=None
        if aa is not None and ab is not None:
            aa=np.asarray(aa,dtype=np.int64)
            ab=np.asarray(ab,dtype=np.int64)
            layer['integer_accumulator_differences']=int(np.count_nonzero(aa!=ab))
            layer['integer_accumulator_max_abs']=int(np.max(np.abs(aa-ab)))
        else:layer['integer_accumulator_differences']=None
        layers.append(layer)
    return layers


class RustClient:
    """One persistent verified Rust process, one normalized observation/request.
    The caller supplies schema identity with every request. No automatic fallback
    to --unsafe-unverified unless explicitly requested for development tests.
    """
    def __init__(self,model_path,rust_binary,*,manifest_path=None,profile_path=None,
                 unsafe_unverified=False):
        model_path=Path(model_path)
        self.schema_hash=None
        cmd=[str(Path(rust_binary).resolve()),'--model',str(model_path.resolve()),'--stdio']
        if manifest_path is not None and profile_path is not None:
            info=verify(read_json(manifest_path),read_json(profile_path),model_path)
            self.schema_hash=info['contract_sha256']
            cmd+=['--manifest',str(Path(manifest_path).resolve()),
                  '--approved-profile',str(Path(profile_path).resolve())]
        elif manifest_path is None and profile_path is None and unsafe_unverified:
            cmd+=['--unsafe-unverified']
        else:
            raise ValueError('Use both manifest/profile, or explicit unsafe_unverified=True in tests')
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,text=True,bufsize=1)
        self.count=0

    def ask(self,row,*,trace=False):
        i=self.count
        self.count+=1
        req={'id':i,'observations':np.asarray(row,dtype=np.float32).astype(float).tolist()}
        if self.schema_hash: req['observation_schema_sha256']=self.schema_hash
        if trace:req['trace']=True
        if self.proc.poll() is not None:raise RuntimeError('Rust process has exited')
        self.proc.stdin.write(json.dumps(req,allow_nan=False)+'\n')
        self.proc.stdin.flush()
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('Rust closed pipe: '+self.proc.stderr.read())
        obj=json.loads(line)
        if obj.get('id')!=i or 'error' in obj:
            raise RuntimeError(f'Rust inference rejected request: {obj}')
        return obj
    
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc.wait()
    def __enter__(self):
        return self
    def __exit__(self,*_):
        self.close()


def _stat(v):
    a=np.asarray(v,dtype=np.float64).ravel()
    return {'max_abs':float(a.max()),'mean_abs':float(a.mean()),
            'p99_abs':float(np.percentile(a,99)),
            'p999_abs':float(np.percentile(a,99.9))}


def _probe(client,model,row,indices,epsilons):
    """Perturb one selected observation component at a time near a failure."""
    out=[]
    for i in indices:
        for eps in epsilons:
            for sign in (-1,1):
                x=np.array(row,dtype=np.float32,copy=True)
                x[i]=np.float32(x[i]+np.float32(sign*eps))
                ref=predict(model,x[None,:])[0]
                response=client.ask(x,trace=True)
                d=np.abs(np.asarray(response['action'],np.float32)-ref)
                nt=numpy_trace(model,x)
                layers=_layer_comparison(nt,response['trace'])
                out.append({'feature':int(i),'requested_delta':float(sign*eps),
                    'actual_delta':float(x[i]-row[i]),'action_error':float(d.max()),
                    'code_differences':[p['quant_code_differences'] for p in layers]})
    return out


def compare(binary,observations,rust_binary,atol,chunk=1000,*,manifest_path=None,
            profile_path=None,diagnose_top=0,fail_report=None,probe_top=0,
            probe_eps=(1e-7,1e-6),probe_indices=(0,1,2,3,4,5),
            unsafe_unverified=False,bounds=None):
    model=load_binary(binary)
    x=np.asarray(observations,dtype=np.float32)
    if x.ndim==1:
        x=x[None,:]
    if not len(x) or x.shape[1]!=model.input_size or not np.isfinite(x).all():
        raise ValueError('invalid observation matrix')
    if atol<0 or not np.isfinite(atol):
        raise ValueError('atol must be finite and nonnegative')
    reference=predict(model,x)
    errors=np.empty((len(x),model.output_size),dtype=np.float64)
    result={'model':str(binary),'variant':model.variant,'samples':len(x),
            'tolerance':float(atol),'verified_deployment':bool(manifest_path and profile_path)}
    with RustClient(binary,rust_binary,manifest_path=manifest_path,profile_path=profile_path,
                    unsafe_unverified=unsafe_unverified) as client:
        for i,row in enumerate(x):
            reply=client.ask(row)
            actual=np.asarray(reply['action'],np.float32)
            if actual.shape!=(model.output_size,):
                raise AssertionError('Rust output shape mismatch')
            errors[i]=np.abs(actual.astype('f8')-reference[i].astype('f8'))
        result.update(_stat(errors))
        result['mae']=result['mean_abs']
        result['max_abs']=result['max_abs']
        per_sample=errors.max(axis=1)
        result['n_over_tolerance']=int(np.count_nonzero(per_sample>atol))
        result['n_over_1e-4']=int(np.count_nonzero(per_sample>1e-4))
        result['n_over_1e-3']=int(np.count_nonzero(per_sample>1e-3))
        result['strict_pass']=bool(result['max_abs']<=atol)
        result['bounded_pass']=False
        if bounds is not None:
            max_allowed,p99_allowed,event_threshold,event_rate_allowed=bounds
            if not (0<=event_rate_allowed<=1 and 0<event_threshold and
                    0<max_allowed and 0<p99_allowed):
                raise ValueError('invalid bounded-error acceptance parameters')
            event_count=int(np.count_nonzero(per_sample>event_threshold))
            event_fraction=float(event_count/len(x))
            result['bounded_acceptance']={
                'max_allowed':float(max_allowed),'p99_allowed':float(p99_allowed),
                'event_threshold':float(event_threshold),
                'event_rate_allowed':float(event_rate_allowed),
                'observed_event_count':event_count,
                'observed_event_fraction':event_fraction}
            result['bounded_pass']=bool(result['max_abs']<=max_allowed and
                result['p99_abs']<=p99_allowed and
                event_fraction<=event_rate_allowed)
        result['pass']=bool(result['strict_pass'] or result['bounded_pass'])
        result['accepted_as']='strict' if result['strict_pass'] else (
            'bounded_offline_only' if result['bounded_pass'] else 'rejected')
        n=max(diagnose_top,probe_top)
        if n:
            worst=np.argsort(per_sample)[-min(n,len(x)):][::-1]
            result['worst_cases']=[]
            for rank,index in enumerate(worst):
                idx=int(index)
                reply=client.ask(x[idx],trace=True)
                if 'trace' not in reply:
                    raise RuntimeError('new Rust binary must support trace')
                nt=numpy_trace(model,x[idx])
                rt=reply['trace']
                record={'index':idx,'max_action_error':float(per_sample[idx]),
                        'observation':x[idx].astype(float).tolist(),
                        'numpy_action':reference[idx].astype(float).tolist(),
                        'rust_action':reply['action'],
                        'layers':_layer_comparison(nt,rt)}
                if rank<diagnose_top:
                    # Preserve actual intermediate values and hexadecimal bit words.
                    record['numpy_trace']=nt
                    record['rust_trace']=rt
                if rank<probe_top:
                    record['boundary_probes']=_probe(client,model,x[idx],
                        [i for i in probe_indices if i<model.input_size],probe_eps)
                result['worst_cases'].append(record)
    if fail_report is not None:
        p=Path(fail_report)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2,allow_nan=False))
    if not result['pass']:
        raise AssertionError(f'Rust / NumPy mismatch (report saved): max={result["max_abs"]:.8g} > {atol}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('observations')
    p.add_argument('--models',required=True)
    p.add_argument('--rust-binary',required=True)
    p.add_argument('--approved-profile',help='Operator-approved deployment profile')
    p.add_argument('--unsafe-unverified',action='store_true',help='Development use only')
    p.add_argument('--max-samples',type=int,default=10000)
    p.add_argument('--variants',nargs='+')
    p.add_argument('--atol',type=float,default=1e-4)
    p.add_argument('--diagnose-top',type=int,default=10)
    p.add_argument('--probe-top',type=int,default=2)
    p.add_argument('--probe-eps',type=float,nargs='+',default=[1e-7,1e-6,1e-5])
    p.add_argument('--probe-indices',type=int,nargs='+',default=list(range(6)))
    p.add_argument('--bound-max',type=float,help='Explicit maximum allowed action error')
    p.add_argument('--bound-p99',type=float,help='Explicit p99 allowed action error')
    p.add_argument('--event-threshold',type=float,help='Action error classified as a boundary event')
    p.add_argument('--bound-event-rate',type=float,help='Maximum allowed fraction of observations exceeding event threshold')
    p.add_argument('--out',default='rust_equivalence_v6.json')
    args=p.parse_args()
    if args.probe_top<0 or args.diagnose_top<0 or min(args.probe_eps)<=0:
        p.error('invalid probe configuration')
    if not args.approved_profile and not args.unsafe_unverified:
        p.error('provide --approved-profile, --unsafe-unverified is for development tests only')
    x=observations_from_file(args.observations)
    if x.ndim==1:
        x=x[None,:]
    if args.max_samples<len(x):
        rng=np.random.default_rng(1107)
        x=x[np.sort(rng.choice(len(x),args.max_samples,replace=False))]
    parts=[args.bound_max,args.bound_p99,args.event_threshold,args.bound_event_rate]
    if any(v is not None for v in parts) and not all(v is not None for v in parts):
        p.error('bounded acceptance requires all four --bound-* / --event-threshold options')
    bounds=tuple(parts) if all(v is not None for v in parts) else None
    results=[]
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    for f in sorted(Path(args.models).glob('*/actor.bin')):
        if args.variants and f.parent.name not in args.variants:
            continue
        manifest=f.parent/'deployment.json'
        if args.approved_profile and not manifest.exists():
            raise ValueError(f'missing manifest: {manifest}')
        per=out.with_name(f'{out.stem}_{f.parent.name}.json')
        try:
            item=compare(f,x,args.rust_binary,args.atol,
                manifest_path=manifest if args.approved_profile else None,
                profile_path=args.approved_profile,unsafe_unverified=args.unsafe_unverified,
                diagnose_top=args.diagnose_top,probe_top=args.probe_top,
                probe_eps=args.probe_eps,probe_indices=args.probe_indices,
                fail_report=per,bounds=bounds)
            results.append({k:v for k,v in item.items() if k!='worst_cases'})
        except AssertionError as exc:
            print(exc)
            item=read_json(per)
            results.append({k:v for k,v in item.items() if k!='worst_cases'})
        print(f'{f.parent.name}: accepted_as={results[-1]["accepted_as"]} '
              f'max={results[-1]["max_abs"]:.9g} n_over={results[-1]["n_over_tolerance"]}, {per}')
    out.write_text(json.dumps(results,indent=2,allow_nan=False))
    if not results:
        raise SystemExit('no variants found')
    if any(not r['pass'] for r in results):
        raise SystemExit(1)



if __name__=='__main__':
    main()
