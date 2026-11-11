from pathlib import Path
import json
import struct
import zlib
import numpy as np
from .core import QuantActor, QuantDense, VARIANTS


MAGIC = b'ADCSMLP1' # ADCSMLP1 versioned little-endian binary wire format, CRC32 trailer.
VERSION = 1
ACTIVATION = {'tanh': 0, 'relu': 1}
MODES = {v: i for i, v in enumerate(VARIANTS)}


def save_binary(model: QuantActor, path):
    if len(model.layers) not in range(1, 17): 
        raise ValueError('invalid layer count')
    data = bytearray(MAGIC)
    data.extend(struct.pack('<III f I', VERSION, MODES[model.variant],
                            ACTIVATION[model.activation], model.mean_limit, len(model.layers)))
    prev = None
    for layer in model.layers:
        n, m = layer.weight.shape
        if prev is not None and prev != n: 
            raise ValueError('layer shape mismatch')
        if not (1 <= n <= 8192 and 1 <= m <= 8192): 
            raise ValueError('dimensions unsafe')
        data.extend(struct.pack('<II', n, m))
        mode = model.variant
        w_dtype = '<f4' if mode == 'fp32' else '<f2' if mode == 'fp16' else '<u2' if mode == 'bf16' else 'i1'
        b_dtype = '<i4' if mode == 'w8a8' else '<f2' if mode == 'fp16' else '<u2' if mode == 'bf16' else '<f4'
        data.extend(np.asarray(layer.weight, dtype=w_dtype).tobytes(order='C'))
        data.extend(np.asarray(layer.bias, dtype=b_dtype).tobytes(order='C'))
        if mode in ('w8a32', 'w8a16', 'w8a8'):
            scales = np.asarray(layer.weight_scales, dtype='<f4')
            if scales.shape != (m,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError('bad weight scales')
            data.extend(scales.tobytes())
            if mode == 'w8a8':
                if layer.input_scale is None or not np.isfinite(layer.input_scale) or layer.input_scale <= 0:
                    raise ValueError('bad input scale')
                data.extend(struct.pack('<f', layer.input_scale))
        prev = m

    data.extend(struct.pack('<I', zlib.crc32(data) & 0xffffffff))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return len(data)


def load_binary(path) -> QuantActor:
    p = Path(path)
    data = p.read_bytes()
    if len(data) > 64 * 1024 * 1024: 
        raise ValueError('model file too large')
    if len(data) < 36 or data[:8] != MAGIC: 
        raise ValueError('bad model header')
    if (zlib.crc32(data[:-4]) & 0xffffffff) != struct.unpack('<I', data[-4:])[0]:
        raise ValueError('CRC32 mismatch')
    
    cursor = 8
    def take(fmt):
        nonlocal cursor
        size = struct.calcsize(fmt)
        if cursor + size > len(data)-4: 
            raise ValueError('truncated model')
        result = struct.unpack_from(fmt, data, cursor)
        cursor += size
        return result
    version, mode, act, limit, layers_count = take('<III f I')
    if version != VERSION or mode >= len(VARIANTS) or act not in (0,1) or not np.isfinite(limit) or limit <= 0 or not (1 <= layers_count <= 16):
        raise ValueError('invalid metadata')
    variant = VARIANTS[mode]
    layers = []
    prev = None
    
    def array(count, dtype):
        nonlocal cursor
        dt = np.dtype(dtype)
        size = count * dt.itemsize
        if cursor + size > len(data)-4: 
            raise ValueError('truncated payload')
        v = np.frombuffer(data, dtype=dt, count=count, offset=cursor).copy()
        cursor += size
        return v
    
    for _ in range(layers_count):
        n, m = take('<II')
        if not (1 <= n <= 8192 and 1 <= m <= 8192) or (prev is not None and n != prev):
            raise ValueError('invalid Dense dimensions')
        wt = '<f4' if variant == 'fp32' else '<f2' if variant == 'fp16' else '<u2' if variant == 'bf16' else 'i1'
        bt = '<i4' if variant == 'w8a8' else '<f2' if variant == 'fp16' else '<u2' if variant == 'bf16' else '<f4'
        w = array(n*m, wt).reshape(n, m)
        b = array(m, bt)
        ws = array(m, '<f4') if variant.startswith('w8') else None
        xs = take('<f')[0] if variant == 'w8a8' else None
        if ws is not None and (not np.isfinite(ws).all() or np.any(ws <= 0)): 
            raise ValueError('bad scales')
        if xs is not None:
            if not np.isfinite(xs) or xs <= 0: 
                raise ValueError('bad scale')
            if np.any(127*127*n + np.abs(b.astype(np.int64)) >= 2**31): 
                raise ValueError('potential int32 overflow')
        layers.append(QuantDense(w, b, ws, xs))
        prev = m
    if cursor != len(data)-4: 
        raise ValueError('unexpected trailing payload')
    
    return QuantActor(layers, variant, ('tanh','relu')[act], float(limit))


def write_metadata(path, model, *, calibration=None, origin=None):
    metadata = {
        'format': 'ADCSMLP1', 'version': VERSION, 'variant': model.variant,
        'activation': model.activation, 'policy_mean_limit': model.mean_limit,
        'input_size': model.input_size, 'output_size': model.output_size,
        'layer_sizes': [model.input_size] + [int(x.weight.shape[1]) for x in model.layers],
        'weights_layout': 'input_major_row_major',
        'action_contract': 'action=tanh(limit*tanh(raw/limit))',
        'w8a16_note': 'FP16 activations, dequantized FP16 weights, FP32 sum',
        'calibration': calibration, 'origin': origin,
        'binary_model': 'actor.bin',
    }

    Path(path).write_text(json.dumps(metadata, indent=2))
    return metadata
