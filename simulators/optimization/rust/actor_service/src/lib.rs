use std::fs;
use std::path::Path;

const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_WIDTH: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Fp32, Fp16, Bf16, W8A32, W8A16, W8A8 }
impl Mode {
    fn from_u32(n:u32)->Result<Self,String> {
        match n {0=>Ok(Self::Fp32),1=>Ok(Self::Fp16),2=>Ok(Self::Bf16),
            3=>Ok(Self::W8A32),4=>Ok(Self::W8A16),5=>Ok(Self::W8A8),
            _=>Err(format!("unsupported precision mode {n}"))}
    }
    pub fn name(&self)->&'static str {
        match self {Self::Fp32=>"fp32",Self::Fp16=>"fp16",Self::Bf16=>"bf16",
            Self::W8A32=>"w8a32",Self::W8A16=>"w8a16",Self::W8A8=>"w8a8"}
    }
}

#[derive(Debug)]
struct Dense {
    n:usize, m:usize,
    float_weights:Vec<f32>, int_weights:Vec<i8>,
    biases_f32:Vec<f32>, biases_i32:Vec<i32>,
    weight_scales:Vec<f32>, input_scale:f32,
}

#[derive(Debug)]
pub struct Actor {
    mode:Mode,
    activation:u32, 
    mean_limit:f32,
    layers:Vec<Dense>,
    input_size:usize, 
    output_size:usize, 
    max_width:usize,
}

pub struct Workspace {
    a:Vec<f32>, 
    b:Vec<f32>, 
    accum:Vec<i32>, 
    qinput:Vec<i8>,
}
impl Workspace {
    pub fn new(model:&Actor)->Self {
        Self { a:vec![0.;model.max_width], b:vec![0.;model.max_width],
            accum:vec![0;model.max_width], qinput:vec![0;model.max_width] }
    }
}

// IEEE754 conversion, round-to-nearest-even.
fn f16_to_f32(bits:u16)->f32 {
    let sign=((bits & 0x8000) as u32)<<16;
    let exponent=((bits>>10)&31) as i32;
    let mant=(bits & 0x3ff) as u32;
    let out=if exponent==0 {
        if mant==0 {sign}
        else {
            let mut m=mant;
            let mut e=-14i32;
            while (m & 0x400)==0 {
                m<<=1;
                e-=1;
            }
            sign | (((e+127) as u32)<<23) | ((m & 0x3ff)<<13)
        }
    } else if exponent==31 {
        sign | 0x7f800000 | (mant<<13)
    }
      else {
        sign | (((exponent-15+127) as u32)<<23) | (mant<<13)
    };
    f32::from_bits(out)
}
fn f32_to_f16(x:f32)->u16 {
    let bits=x.to_bits();
    let sign=((bits>>16)&0x8000) as u16;
    let e=((bits>>23)&0xff) as i32;
    let mant=bits & 0x7fffff;
    if e==255 {
        return sign|0x7c00|if mant!=0 {0x0200} else {0};
    }
    let he=e-127+15;
    if he>=31 {
        return sign|0x7c00;
    }
    if he<=0 {
        if he < -10 {
            return sign;
        }
        let m=mant|0x800000;
        let shift=(14-he) as u32;
        let base=m>>shift;
        let remainder=m & ((1u32<<shift)-1);
        let halfway=1u32<<(shift-1);
        let rounded=base+u32::from(remainder>halfway || (remainder==halfway && (base&1)!=0));
        return sign|(rounded as u16);
    }
    let base=mant>>13;
    let rem=mant & 0x1fff;
    let rounded=base+u32::from(rem>0x1000 || (rem==0x1000 && (base&1)!=0));
    let mut exp=he as u16;
    let mut fraction=rounded as u16;
    if rounded==1024 {
        exp+=1;fraction=0;
    }
    if exp>=31 {
        return sign|0x7c00;
    }
    sign|(exp<<10)|(fraction&0x03ff)
}

fn to_fp16_f32(x:f32)->f32 {
    f16_to_f32(f32_to_f16(x))
}

fn to_bf16_f32(x:f32)->f32 {
    let bits=x.to_bits();
    let rounded=bits.wrapping_add(0x7fff + ((bits>>16)&1));
    f32::from_bits(rounded & 0xffff0000)
}

fn round8(x:f32)->i8 {
    x.round_ties_even().clamp(-127.,127.) as i8
}

fn crc32(data:&[u8])->u32 {
    let mut crc=0xffffffffu32;
    for b in data {
        crc^=*b as u32;
        for _ in 0..8 {
            crc=(crc>>1)^if crc&1!=0 {0xedb88320} else {0};
        }
    }
    !crc
}

struct Reader<'a>{
    bytes:&'a [u8],pos:usize
}

impl<'a> Reader<'a>{
    fn take(&mut self,n:usize)->Result<&'a [u8],String>{
        if n>self.bytes.len().saturating_sub(self.pos) {
            return Err("truncated binary model".into());
        }
        let result=&self.bytes[self.pos..self.pos+n];self.pos+=n;
        Ok(result)
    }

    fn u32(&mut self)->Result<u32,String>{
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self)->Result<i32,String>{
        Ok(self.u32()? as i32)
    }
    fn f32(&mut self)->Result<f32,String>{
        Ok(f32::from_bits(self.u32()?))
    }
    fn read_float(&mut self,mode:Mode)->Result<f32,String>{
        match mode {
            Mode::Fp16=>Ok(f16_to_f32(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))),
            Mode::Bf16=>Ok(f32::from_bits((u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u32)<<16)),
            _=>self.f32(),
        }
    }
}
impl Actor {
    pub fn load(path:impl AsRef<Path>)->Result<Self,String>{
        let bytes=fs::read(path).map_err(|e|e.to_string())?;
        Self::from_bytes(&bytes)
    }
    pub fn from_bytes(bytes:&[u8])->Result<Self,String>{
        if bytes.len()>MAX_BYTES {
            return Err("model binary too large".into());
        }
        if bytes.len()<36 || &bytes[0..8]!=b"ADCSMLP1" {
            return Err("invalid ADCSMLP1 magic".into());
        }
        let expected=u32::from_le_bytes(bytes[bytes.len()-4..].try_into().unwrap());
        if crc32(&bytes[..bytes.len()-4])!=expected {
            return Err("model CRC32 mismatch".into());
        }

        let mut r=Reader{bytes:&bytes[..bytes.len()-4],pos:8};
        if r.u32()?!=1 {
            return Err("unsupported format version".into());
        }
        let mode=Mode::from_u32(r.u32()?)?;
        let activation=r.u32()?;
        if activation>1 {
            return Err("invalid activation".into());
        }
        let mean_limit=r.f32()?;
        if !mean_limit.is_finite() || mean_limit<=0. {
            return Err("invalid mean limit".into());
        }
        let count=r.u32()? as usize;
        if count==0 || count>16 {
            return Err("invalid layer count".into());
        }
        let mut prev=None;
        let mut layers=Vec::with_capacity(count);
        let mut max_width=0;

        for _ in 0..count {
            let n=r.u32()? as usize;
            let m=r.u32()? as usize;
            if n==0 || m==0 || n>MAX_WIDTH || m>MAX_WIDTH || prev.is_some_and(|p|p!=n) {
                return Err("invalid Dense dimensions".into());
            }
            let size=n.checked_mul(m).ok_or("invalid dimensions")?;
            let mut float_weights=Vec::new();
            let mut int_weights=Vec::new();
            if matches!(mode,Mode::Fp32|Mode::Fp16|Mode::Bf16) {
                float_weights.reserve(size);
                for _ in 0..size {
                    float_weights.push(r.read_float(mode)?);
                }
            } else {
                int_weights.extend(r.take(size)?.iter().map(|&x|x as i8));
            }
            let mut biases_f32=Vec::new();
            let mut biases_i32=Vec::new();
            if mode==Mode::W8A8 {
                for _ in 0..m {
                    biases_i32.push(r.i32()?);
                }
            } else {
                for _ in 0..m {
                    biases_f32.push(r.read_float(mode)?);
                }
            }
            let mut weight_scales=Vec::new(); 
            let mut input_scale=0f32;
            if matches!(mode,Mode::W8A32|Mode::W8A16|Mode::W8A8) {
                for _ in 0..m {
                    weight_scales.push(r.f32()?);
                }
                if weight_scales.iter().any(|x|!x.is_finite() || *x<=0.) {
                    return Err("invalid weight scale".into());
                }
                if mode==Mode::W8A8 {
                    input_scale=r.f32()?;
                    if !input_scale.is_finite() || input_scale<=0. {
                        return Err("invalid input scale".into());
                    }
                    let max_product=(n as i64)*127*127;
                    if biases_i32.iter().any(|b|max_product+(*b as i64).abs() >= (1i64<<31)) {
                        return Err("INT32 overflow risk".into());
                    }
                }
            }
            if float_weights.iter().any(|v|!v.is_finite()) || biases_f32.iter().any(|v|!v.is_finite()) {
                return Err("non-finite parameters".into());
            }
            max_width=max_width.max(n).max(m);
            layers.push(Dense{n,m,float_weights,int_weights,biases_f32,biases_i32,weight_scales,input_scale});
            prev=Some(m);
        }
        if r.pos!=r.bytes.len() {return Err("unexpected bytes after layers".into());}
        Ok(Self{mode,activation,mean_limit,
            input_size:layers[0].n,output_size:layers.last().unwrap().m,max_width,layers})
    }

    pub fn mode(&self)->Mode{self.mode}
    pub fn input_size(&self)->usize{self.input_size}
    pub fn output_size(&self)->usize{self.output_size}
    pub fn layers(&self)->usize{self.layers.len()}

    pub fn predict_into(&self, obs:&[f32], ws:&mut Workspace,mean:&mut[f32],action:&mut[f32])->Result<(),String>{
        if obs.len()!=self.input_size || mean.len()!=self.output_size || action.len()!=self.output_size {
            return Err("observation/output dimensions do not match model".into());
        }
        if obs.iter().any(|x|!x.is_finite()) {
            return Err("non-finite observation".into());
        }
        ws.a[..obs.len()].copy_from_slice(obs);
        let mut width=obs.len();
        for (li,layer) in self.layers.iter().enumerate(){
            debug_assert_eq!(width,layer.n);
            let m=layer.m;
            match self.mode {
                Mode::W8A8=>{
                    for i in 0..width {ws.qinput[i]=round8(ws.a[i]/layer.input_scale);}
                    ws.accum[..m].copy_from_slice(&layer.biases_i32);
                    for i in 0..width {
                        let qx=ws.qinput[i] as i32;
                        for j in 0..m {
                            // Bound validated at model loading time.
                            ws.accum[j]+=qx*(layer.int_weights[i*m+j] as i32);
                        }
                    }
                    for j in 0..m {
                        ws.b[j]=(ws.accum[j] as f32)*(layer.input_scale*layer.weight_scales[j]);
                    }
                }
                Mode::W8A32|Mode::W8A16=>{
                    ws.b[..m].copy_from_slice(&layer.biases_f32);
                    for i in 0..width {
                        let x = if self.mode==Mode::W8A16 {to_fp16_f32(ws.a[i])} else {ws.a[i]};
                        for j in 0..m {
                            let mut w=(layer.int_weights[i*m+j] as f32)*layer.weight_scales[j];
                            if self.mode==Mode::W8A16 {
                                w=to_fp16_f32(w);
                            }
                            ws.b[j]+=x*w;
                        }
                    }
                    if self.mode==Mode::W8A16 {
                        for j in 0..m {
                            ws.b[j]=to_fp16_f32(ws.b[j]);
                        }
                    }
                }
                Mode::Fp32|Mode::Fp16|Mode::Bf16=>{
                    ws.b[..m].copy_from_slice(&layer.biases_f32);
                    for i in 0..width {
                        let x=match self.mode {
                            Mode::Fp16=>to_fp16_f32(ws.a[i]),
                            Mode::Bf16=>to_bf16_f32(ws.a[i]),
                            _=>ws.a[i],
                        };
                        for j in 0..m {
                            ws.b[j]+=x*layer.float_weights[i*m+j];
                        }
                    }
                    match self.mode {
                        Mode::Fp16=>for j in 0..m {
                            ws.b[j]=to_fp16_f32(ws.b[j]);
                        },
                        Mode::Bf16=>for j in 0..m {
                            ws.b[j]=to_bf16_f32(ws.b[j]);
                        },
                        _=>{},
                    }
                }
            }
            if li+1!=self.layers.len(){
                for j in 0..m {
                    let x = match self.activation {
                        0=> ws.b[j].tanh(),_=>ws.b[j].max(0.)
                    };
                    ws.a[j]=match self.mode {
                        Mode::Fp16|Mode::W8A16=>to_fp16_f32(x),
                        Mode::Bf16=>to_bf16_f32(x),
                        _=>x,
                    };
                }
            } else {
                for j in 0..m {
                    mean[j] = self.mean_limit*(ws.b[j]/self.mean_limit).tanh();
                    action[j] = mean[j].tanh();
                    if !mean[j].is_finite() || !action[j].is_finite() {
                        return Err("non-finite inference result".into());
                    }
                }
            }
            width = m;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn known_f16_values() {
        for x in [0.,1.,-1.,0.5,0.1,3.,1e-5,65504.] {
            let y=f16_to_f32(f32_to_f16(x));
            assert!((y-x).abs()<=x.abs()*0.001+0.00001);
        }
    }
    #[test] fn invalid_crc() {
        let mut data=b"ADCSMLP1\x01\0\0\0".to_vec();data.extend([0;20]);
        assert!(Actor::from_bytes(&data).is_err());
    }
    #[test] fn bfloat_round_trip() {
        assert_eq!(to_bf16_f32(1.0),1.0);
        assert_eq!(to_fp16_f32(0.5),0.5);
    }
}
