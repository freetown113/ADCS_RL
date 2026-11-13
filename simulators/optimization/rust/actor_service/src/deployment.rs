use std::{fs,path::Path};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest,Sha256};
use adcs_actor_service::Actor;


#[derive(Debug,Deserialize)]
struct BinaryRef { 
    sha256:String, 
    bytes:usize 
}

#[derive(Debug,Deserialize)]
struct ModelInfo {
    variant:String,
    input_size:usize,
    output_size:usize,
    layer_sizes:Vec<usize>,
    activation:String,
    policy_mean_limit:f32,
    action_contract:String,
    origin_actor_sha256:String,
}

#[derive(Debug,Deserialize)]
struct Manifest {
    kind:String,
    version:u32,
    contract:Value,
    contract_sha256:String,
    binary:BinaryRef,
    model:ModelInfo,
}

#[derive(Debug,Deserialize)]
struct Approval {
    kind:String,
    version:u32,
    contract:Value,
    contract_sha256:String,
    allowed_actor_sha256:Vec<String>,
    allowed_variants:Vec<String>,
    allowed_binary_sha256:Vec<String>,
}

pub struct VerifiedContract {pub hash:String}
fn load_json<T:for<'de>Deserialize<'de>>(path:&str)->Result<T,String> {
    let b=fs::read(path).map_err(|e|format!("cannot read {path}: {e}"))?;
    if b.len()>1024*1024 {return Err(format!("contract JSON too large: {path}"));}
    serde_json::from_slice(&b).map_err(|e|format!("invalid JSON {path}: {e}"))
}

pub fn verify(model_path:&str,manifest_path:&str,approval_path:&str,actor:&Actor)->Result<VerifiedContract,String>{
    let mf:Manifest=load_json(manifest_path)?;
    let ap:Approval=load_json(approval_path)?;
    if mf.kind!="adcs.actor.deployment.v1" || mf.version!=1 ||
       ap.kind!="adcs.actor.approved-profile.v1" || ap.version!=1 {
        return Err("unsupported deployment manifest / profile schema".into());
    }

    if mf.contract!=ap.contract || mf.contract_sha256!=ap.contract_sha256 || mf.contract_sha256.len()!=64 {
        return Err("observation schema / normalization / output contract mismatch".into());
    }
    if !ap.allowed_actor_sha256.contains(&mf.model.origin_actor_sha256) ||
       !ap.allowed_variants.contains(&mf.model.variant) ||
       !ap.allowed_binary_sha256.contains(&mf.binary.sha256) {
        return Err("actor hash, binary hash or precision variant not approved by deployment profile".into());
    }
    let bytes=fs::read(Path::new(model_path)).map_err(|e|e.to_string())?;
    let hash=format!("{:x}",Sha256::digest(&bytes));
    if bytes.len()!=mf.binary.bytes || hash!=mf.binary.sha256 {
        return Err("model binary SHA-256 or file size mismatch".into());
    }
    let sizes=actor.layer_sizes();
    if mf.model.input_size!=actor.input_size() || mf.model.output_size!=actor.output_size() ||
       mf.model.layer_sizes!=sizes || mf.model.variant!=actor.mode().name() ||
       mf.model.activation!=actor.activation_name() ||
       (mf.model.policy_mean_limit-actor.mean_limit()).abs()>1e-6 ||
       mf.model.action_contract!="action=tanh(limit*tanh(raw/limit))" {
        return Err("model binary disagrees with signed-off manifest architecture".into());
    }
    let cin=mf.contract.get("input_size").and_then(Value::as_u64);
    let cout=mf.contract.get("output_size").and_then(Value::as_u64);
    if cin!=Some(actor.input_size() as u64) || cout!=Some(actor.output_size() as u64) {
        return Err("observation contract dimensions do not match actor".into());
    }
    Ok(VerifiedContract {hash:mf.contract_sha256})
}
