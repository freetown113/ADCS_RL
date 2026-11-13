use std::env;
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpListener;
use std::time::Instant;
use std::sync::Arc;
use serde::{Deserialize,Serialize};
use serde_json::Value;
use adcs_actor_service::{Actor,Workspace};

#[derive(Deserialize)]
struct Request {
    #[serde(default)] id:Option<Value>,observations:Vec<f32>
}
#[derive(Serialize)]
struct Response<'a> {
    #[serde(skip_serializing_if="Option::is_none")] id:Option<Value>,action:&'a [f32],mean:&'a [f32]
}
#[derive(Serialize)]
struct Failure {
    #[serde(skip_serializing_if="Option::is_none")] id:Option<Value>,error:String
}

fn serve<R:BufRead,W:Write>(input:R,mut output:W,model:&Actor)->io::Result<()> {
    let mut ws=Workspace::new(model);
    let mut mean=vec![0f32;model.output_size()];
    let mut action=vec![0f32;model.output_size()];
    for line in input.lines() {
        let line=line?;
        if line.trim().is_empty(){
            continue;
        }
        let parsed=serde_json::from_str::<Request>(&line);
        let result=match parsed {
            Ok(req)=>{
                match model.predict_into(&req.observations,&mut ws,&mut mean,&mut action){
                    Ok(())=>serde_json::to_string(&Response{id:req.id,action:&action,mean:&mean}),
                    Err(err)=>serde_json::to_string(&Failure{id:req.id,error:err}),
                }
            }
            Err(err)=>serde_json::to_string(&Failure{id:None,error:format!("invalid JSON request: {err}")}),
        };
        writeln!(output,"{}",result.map_err(io::Error::other)?)?;
        output.flush()?;
    }
    Ok(())
}
fn bench(model:&Actor,n:usize)->Result<(),String>{
    if n==0{
        return Err("benchmark count must be positive".into());
    }
    let obs=vec![0f32;model.input_size()];
    let mut ws=Workspace::new(model);
    let mut mean=vec![0f32;model.output_size()];
    let mut action=vec![0f32;model.output_size()];
    for _ in 0..1000 {
        model.predict_into(&obs,&mut ws,&mut mean,&mut action)?;
    }
    let mut times=Vec::with_capacity(n);
    for _ in 0..n {
        let start=Instant::now();
        model.predict_into(&obs,&mut ws,&mut mean,&mut action)?;
        std::hint::black_box(&action);
        times.push(start.elapsed().as_secs_f64()*1e6);
    }
    times.sort_by(|a,b|a.total_cmp(b));
    println!("{{\"mode\":\"{}\",\"iterations\":{},\"p50_us\":{},\"p99_us\":{},\"max_us\":{}}}",
        model.mode().name(),n,times[n/2],times[((n-1)*99)/100],times[n-1]);
    Ok(())
}
fn run()->Result<(),String>{
    let mut model_path=None;
    let mut listen=None;
    let mut repeats=None;
    let mut it=env::args().skip(1);
    while let Some(arg)=it.next(){
        match arg.as_str(){
            "--model"=>model_path=it.next(),
            "--listen"=>listen=it.next(),
            "--bench"=>repeats=Some(it.next().ok_or("missing --bench count")?.parse::<usize>().map_err(|e|e.to_string())?),
            "--stdio"=>{},
            "--help"|"-h"=>{
                println!("Usage: adcs-actor-service --model actor.bin [--stdio | --listen 127.0.0.1:7777 | --bench N]");return Ok(());
            },
            _=>return Err(format!("unknown argument {arg}")),
        }
    }
    let path=model_path.ok_or("--model is required")?;
    let model=Arc::new(Actor::load(path)?);
    eprintln!("ADCS actor {}: {} inputs -> {} outputs, {} layers",model.mode().name(),model.input_size(),model.output_size(),model.layers());
    
    if let Some(n)=repeats {
        return bench(&model,n);
    }
    if let Some(addr)=listen {
        // Opt-in TCP mode: bind loopback only (no public network exposure).
        if !addr.starts_with("127.0.0.1:") && !addr.starts_with("[::1]:") {
            return Err("--listen only supports localhost addresses".into());
        }
        let socket=TcpListener::bind(&addr).map_err(|e|e.to_string())?;
        eprintln!("NDJSON TCP listening on {addr}");
        for client in socket.incoming(){
            let socket=client.map_err(|e|e.to_string())?;
            let model=model.clone();
            // One connection at a time: bounded resource use, no task scheduling jitter.
            let input=BufReader::new(socket.try_clone().map_err(|e|e.to_string())?);
            if let Err(e)=serve(input,socket,&model){
                eprintln!("client error: {e}");
            }
        }
    } else {
        let stdin=io::stdin();
        let stdout=io::stdout();
        serve(stdin.lock(),stdout.lock(),&model).map_err(|e|e.to_string())?;
    }
    Ok(())
}
fn main(){
    if let Err(err)=run(){
        eprintln!("error: {err}");std::process::exit(1);
    }
}
