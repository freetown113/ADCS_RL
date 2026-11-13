import argparse
import hashlib
import json
from pathlib import Path
from .binary import load_binary


KIND = 'adcs.actor.deployment.v1'
PROFILE_KIND = 'adcs.actor.approved-profile.v1'
CONTRACT_KIND = 'adcs.actor.observation-contract.v1'
ACTION = 'action=tanh(limit*tanh(raw/limit))'


def canonical_bytes(v):
    return json.dumps(v,sort_keys=True,separators=(',', ':'),ensure_ascii=True,allow_nan=False).encode('utf-8')

def sha_value(v):
    return hashlib.sha256(canonical_bytes(v)).hexdigest()

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path,obj):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')

def validate_contract(c):
    if c.get('kind') != CONTRACT_KIND or c.get('version') != 1:
        raise ValueError('invalid contract kind/version')
    n=c.get('input_size')
    m=c.get('output_size')
    if not isinstance(n,int) or not isinstance(m,int) or n<1 or m<1:
        raise ValueError('invalid input/output sizes')
    features=c.get('features')
    if not isinstance(features,list) or len(features)!=n or [e.get('index') for e in features]!=list(range(n)):
        raise ValueError('feature order/index must cover every input exactly')
    for f in features:
        if any(not isinstance(f.get(k),str) or not f[k].strip() for k in ('name','source','units','transform')):
            raise ValueError('feature must specify name/source/units/transform')
    if not isinstance(c.get('producer_schema'),str) or not c['producer_schema']:
        raise ValueError('missing producer schema')
    if c.get('control_mode') not in ('motor_direct','body_torque','residual'):
        raise ValueError('unknown control mode')
    action=c.get('action')
    if not isinstance(action,dict) or action.get('size')!=m or action.get('network_transform')!=ACTION:
        raise ValueError('unsupported action contract')
    if len(action.get('channels',[]))!=m or not action.get('interpretation'):
        raise ValueError('missing action channel mapping/interpretation')
    if not isinstance(c.get('observation_config'),dict):
        raise ValueError('missing observation config')
    return c

def current_contract(calibration,attitude_source,wheel_status_source=None):
    base=calibration.get('observation_contract')
    if not isinstance(base,dict):
        raise ValueError('calibration has no observation_contract, supply a manually audited contract')
    if (base.get('observation_size'),base.get('action_size'),base.get('control_mode'))!=(18,4,'motor_direct'):
        raise ValueError('automatic contract generation supports 18-input/4-output motor_direct only')
    cfg=base.get('observation_config',{})
    if not (cfg.get('include_previous_action') and cfg.get('include_wheel_mask')) or cfg.get('include_estimator_confidence') or cfg.get('include_supervisory_mode'):
        raise ValueError('unsupported observation layout, create a custom audited contract')
    if attitude_source not in ('mekf','truth-regression'):
        raise ValueError('explicit attitude source required: mekf or truth-regression')
    mask=cfg.get('wheel_mask_source',wheel_status_source)
    if not mask or (wheel_status_source and cfg.get('wheel_mask_source') and wheel_status_source!=mask):
        raise ValueError('wheel mask source must be supplied or match calibration config')
    omega=base.get('omega_scale')
    wheel=base.get('max_wheel_speed')
    if not isinstance(omega,(int,float)) or not isinstance(wheel,(int,float)) or omega<=0 or wheel<=0:
        raise ValueError('missing physical normalization parameters')
    f=[]
    for j in range(3): f.append({'index':j,'name':f'attitude_error_vector_{j}','source':attitude_source,'units':'dimensionless','transform':'2 * attitude_error_quaternion.xyz (estimated target-relative)'})
    for j in range(3): f.append({'index':j+3,'name':f'body_tracking_rate_error_{j}','source':attitude_source,'units':'normalized rad/s','transform':f'(omega_body - target_rate_body) / {omega}'})
    for j in range(4): f.append({'index':j+6,'name':f'wheel_speed_{j}','source':'wheel_speed_sensor','units':'normalized rad/s','transform':f'measured wheel speed / {wheel}'})
    for j in range(4): f.append({'index':j+10,'name':f'previous_motor_action_{j}','source':'previous_action','units':'normalized motor command','transform':'unchanged previous normalized policy action'})
    for j in range(4): f.append({'index':j+14,'name':f'wheel_authority_{j}','source':mask,'units':'dimensionless','transform':'wheel authority/mask from configured source'})
    c={'kind':CONTRACT_KIND,'version':1,'producer_schema':base.get('schema','unspecified'),
       'input_size':18,'output_size':4,'features':f,'control_mode':'motor_direct',
       'observation_config':cfg,'normalization':{'omega_scale_rad_s':omega,'max_wheel_speed_rad_s':wheel},
       'attitude_source':attitude_source,'wheel_status_source':mask,
       'action':{'size':4,'network_transform':ACTION,
                 'channels':[f'reaction_wheel_motor_{i}' for i in range(4)],
                 'interpretation':'normalized per-wheel motor commands, allocation, limits and FDIR outside actor'}}
    return validate_contract(c)

def seal(model_path,model_json_path,contract_path):
    mdl=load_binary(model_path)
    md=read_json(model_json_path)
    c=validate_contract(read_json(contract_path))
    if (md.get('variant'),md.get('input_size'),md.get('output_size'))!=(mdl.variant,mdl.input_size,mdl.output_size):
        raise ValueError('model.json metadata does not match serialized actor')
    if (mdl.input_size,mdl.output_size)!=(c['input_size'],c['output_size']):
        raise ValueError('model dimensions not compatible with contract')
    if (md.get('action_contract')!=ACTION or md.get('activation')!=mdl.activation or
        abs(float(md.get('policy_mean_limit',0))-mdl.mean_limit)>1e-5):
        raise ValueError('model action postprocessing mismatch')
    observed=md.get('calibration',{})
    observed=observed or {}
    origin=md.get('origin',{}) or {}
    collected=observed.get('observation_contract')
    if collected:
        if (collected.get('observation_config')!=c['observation_config'] or
            collected.get('omega_scale')!=c['normalization']['omega_scale_rad_s'] or
            collected.get('max_wheel_speed')!=c['normalization']['max_wheel_speed_rad_s'] or
            collected.get('control_mode')!=c['control_mode']):
            raise ValueError('calibration observation contract conflicts with requested deployment contract')
    actor_hash=origin.get('actor_sha256')
    if not actor_hash or len(actor_hash)!=64:
        raise ValueError('model.json missing trusted origin.actor_sha256, regenerate export')
    b=Path(model_path).read_bytes()
    manifest={'kind':KIND,'version':1,'contract':c,'contract_sha256':sha_value(c),
              'binary':{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)},
              'model':{'variant':mdl.variant,'input_size':mdl.input_size,'output_size':mdl.output_size,
                       'layer_sizes':[mdl.input_size]+[d.weight.shape[1] for d in mdl.layers],
                       'activation':mdl.activation,'policy_mean_limit':mdl.mean_limit,
                       'action_contract':ACTION,'origin_actor_sha256':actor_hash,
                       'calibration_sha256':origin.get('calibration_sha256')}}
    return manifest

def approve(contract,actor_sha256,variants,binary_hashes):
    c=validate_contract(contract)
    if len(actor_sha256)!=64 or any(k not in '0123456789abcdef' for k in actor_sha256):
        raise ValueError('actor sha256 must be a lowercase 64-hex value')
    if not variants:raise ValueError('at least one approved variant required')
    if not binary_hashes or any(len(v)!=64 or any(k not in '0123456789abcdef' for k in v) for v in binary_hashes):
        raise ValueError('explicit SHA256 digests of reviewed binary artifacts required')
    return {'kind':PROFILE_KIND,'version':1,'contract':c,'contract_sha256':sha_value(c),
            'allowed_actor_sha256':[actor_sha256],'allowed_variants':list(variants),
            'allowed_binary_sha256':list(binary_hashes)}

def verify(manifest,profile,model_path):
    if manifest.get('kind')!=KIND or manifest.get('version')!=1 or profile.get('kind')!=PROFILE_KIND or profile.get('version')!=1:
        raise ValueError('manifest/approval kind or version mismatch')
    c=validate_contract(manifest['contract'])
    approved=validate_contract(profile['contract'])
    if c!=approved or sha_value(c)!=manifest.get('contract_sha256') or sha_value(approved)!=profile.get('contract_sha256'):
        raise ValueError('incompatible observation/action contract or schema hash')
    m=manifest['model']
    b=manifest['binary']
    if (m['origin_actor_sha256'] not in profile['allowed_actor_sha256'] or
        m['variant'] not in profile['allowed_variants'] or
        b['sha256'] not in profile.get('allowed_binary_sha256',[])):
        raise ValueError('actor or quantization variant is not approved')
    if Path(model_path).stat().st_size!=b['bytes'] or sha_file(model_path)!=b['sha256']:
        raise ValueError('binary model sha256/length mismatch')
    actor=load_binary(model_path)
    if (actor.variant,actor.input_size,actor.output_size)!=(m['variant'],m['input_size'],m['output_size']):
        raise ValueError('binary model header mismatch')
    if (actor.input_size,actor.output_size)!=(c['input_size'],c['output_size']):
        raise ValueError('binary shape and contract mismatch')
    if [actor.input_size]+[d.weight.shape[1] for d in actor.layers]!=m['layer_sizes'] or actor.activation!=m['activation'] or abs(actor.mean_limit-m['policy_mean_limit'])>1e-6 or m['action_contract']!=ACTION:
        raise ValueError('binary model topology/nonlinearities mismatch')
    return {'verified':True,'mode':actor.variant,'contract_sha256':sha_value(c),'binary_sha256':b['sha256']}


def main(argv=None):
    ap=argparse.ArgumentParser(description='Versioned ADCS actor deployment contract generator/verifier')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('make-contract')
    p.add_argument('--calibration',required=True)
    p.add_argument('--attitude-source',required=True,choices=('mekf','truth-regression'))
    p.add_argument('--wheel-status-source')
    p.add_argument('--out',required=True)
    p=sub.add_parser('seal')
    p.add_argument('--model',required=True)
    p.add_argument('--model-json',required=True)
    p.add_argument('--contract',required=True)
    p.add_argument('--out',required=True)
    p=sub.add_parser('approve')
    p.add_argument('--contract',required=True)
    p.add_argument('--actor-sha256',required=True)
    p.add_argument('--variants',nargs='+',required=True)
    p.add_argument('--binary-sha256',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p=sub.add_parser('verify')
    p.add_argument('--model',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--approved-profile',required=True)
    args=ap.parse_args(argv)
    if args.command=='make-contract':obj=current_contract(read_json(args.calibration),args.attitude_source,args.wheel_status_source)
    elif args.command=='seal':obj=seal(args.model,args.model_json,args.contract)
    elif args.command=='approve':obj=approve(read_json(args.contract),args.actor_sha256,args.variants,args.binary_sha256)
    else:
        print(json.dumps(verify(read_json(args.manifest),read_json(args.approved_profile),args.model),indent=2))
        return
    write_json(args.out,obj)
    print(f'{args.command}: {args.out}')



if __name__=='__main__':
    main()
