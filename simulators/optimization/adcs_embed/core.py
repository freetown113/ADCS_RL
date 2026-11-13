from dataclasses import dataclass
from typing import Mapping, Any
import numpy as np


VARIANTS = ("fp32", "fp16", "bf16", "w8a32", "w8a16", "w8a8")

@dataclass
class Dense:
    weight: np.ndarray
    bias: np.ndarray

@dataclass
class Actor:
    layers: list[Dense]
    activation: str = "tanh"
    mean_limit: float = 2.0

    def validate(self) -> None:
        if not self.layers or len(self.layers) > 16:
            raise ValueError("actor needs 1..16 layers")
        if self.activation not in ("tanh", "relu"):
            raise ValueError("unsupported activation")
        if not (np.isfinite(self.mean_limit) and self.mean_limit > 0):
            raise ValueError("invalid policy_mean_limit")
        last = None
        for layer in self.layers:
            w, b = np.asarray(layer.weight), np.asarray(layer.bias)
            if w.ndim != 2 or b.shape != (w.shape[1],):
                raise ValueError("incorrect weight/bias dimensions")
            if not (1 <= w.shape[0] <= 8192 and 1 <= w.shape[1] <= 8192):
                raise ValueError("unsupported layer dimensions")
            if last is not None and w.shape[0] != last:
                raise ValueError("disconnected Dense layers")
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError("nonfinite parameters")
            last = w.shape[1]

    @property
    def input_size(self):
        return self.layers[0].weight.shape[0]

    @property
    def output_size(self):
        return self.layers[-1].weight.shape[1]


def _find(params: Mapping[str, Any], key: str):
    matches = [v for k, v in params.items() if k == key or k.endswith('/' + key) or k.endswith('~' + key)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Haiku module {key!r}; found {len(matches)}")
    return matches[0]


def extract_haiku_actor(params: Mapping[str, Any], *, hidden_count: int,
                        activation="tanh", mean_limit=2.0) -> Actor:
    layers = []
    for key in [*(f'actor_hidden_{i}' for i in range(hidden_count)), 'policy_mean']:
        mod = _find(params, key)
        layers.append(Dense(np.asarray(mod['w'], np.float32).copy(),
                            np.asarray(mod['b'], np.float32).copy()))
    a = Actor(layers, activation, float(mean_limit))
    a.validate()
    return a


def _round_bf16(x):
    u = np.asarray(x, dtype='<f4').view('<u4')
    # Add 0x7fff + current lower mantissa's top retained LSB.
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype('<u2').view('<u2')


def _bf16_to_f32(words):
    return (np.asarray(words, dtype='<u2').astype('<u4') << 16).view('<f4')


def cast_to_bf16(x):
    return _bf16_to_f32(_round_bf16(x))


@dataclass
class QuantDense:
    weight: np.ndarray                # fp32, fp16, uint16 bf16, or int8
    bias: np.ndarray                  # fp32/fp16/bf16 words/int32
    weight_scales: np.ndarray | None = None
    input_scale: float | None = None

@dataclass
class QuantActor:
    layers: list[QuantDense]
    variant: str
    activation: str
    mean_limit: float

    @property
    def input_size(self): return self.layers[0].weight.shape[0]
    @property
    def output_size(self): return self.layers[-1].weight.shape[1]


def _round_int8(x, scale):
    # np.rint = ties-to-even, as in Rust f32::round_ties_even.
    return np.clip(np.rint(np.asarray(x, np.float32) / np.float32(scale)), -127, 127).astype(np.int8)


def _weight_quant(w):
    w = np.asarray(w, np.float32)
    scales = np.maximum(np.max(np.abs(w), axis=0), np.float32(1.e-12)) / np.float32(127.0)
    q = _round_int8(w, scales[None, :])
    return q, scales.astype(np.float32)


def _hidden(x, activation):
    return np.tanh(x) if activation == 'tanh' else np.maximum(x, 0.)


def float_hidden_outputs(actor: Actor, obs: np.ndarray):
    """FP32 intermediate activations (for ReLU calibration diagnostics)."""
    x = np.asarray(obs, np.float32)
    for layer in actor.layers[:-1]:
        x = _hidden(x @ layer.weight + layer.bias, actor.activation).astype(np.float32)
        yield x


def calibrate(actor: Actor, observations: np.ndarray, *, percentile=100., margin=1.05,
              chunk_size=65536, max_samples=None):
    """Static per-tensor activation ranges. No held-out evaluation data allowed.
    Hidden tanh uses guaranteed [-1,1], ReLU hidden uses representative outputs.
    Uses an absolute percentile across all tensor elements.
    """
    if not (0. < percentile <= 100.) or margin < 1.:
        raise ValueError('percentile must be (0,100], margin >= 1')
    obs = np.asarray(observations, np.float32)
    if obs.ndim != 2 or obs.shape[1] != actor.input_size or not len(obs):
        raise ValueError('invalid observation batch')
    if max_samples is not None: obs = obs[:max_samples]
    if not np.isfinite(obs).all(): raise ValueError('non-finite observations')
    def calc(x):
        bound = float(np.percentile(np.abs(x), percentile)) * margin
        return max(bound, 1.e-8) / 127.
    input_scale = calc(obs)
    if actor.activation == 'tanh':
        scales = [input_scale] + [1. / 127.] * (len(actor.layers) - 1)
    else:
        vals = [[] for _ in actor.layers[:-1]]
        for i in range(0, len(obs), chunk_size):
            for j, v in enumerate(float_hidden_outputs(actor, obs[i:i+chunk_size])):
                vals[j].append(v)
        scales = [input_scale] + [calc(np.concatenate(v, axis=0)) for v in vals]
    frac_clip = float(np.mean(np.abs(obs) > input_scale * 127))
    return {'input_scales': [float(s) for s in scales],
            'calibration_rows': int(len(obs)), 'percentile': float(percentile),
            'margin': float(margin), 'input_clip_fraction': frac_clip,
            'observed_absmax': float(np.max(np.abs(obs))),
            'per_feature_absmax': np.max(np.abs(obs),axis=0).astype(float).tolist(),
            'per_feature_clip_fraction': np.mean(np.abs(obs)>input_scale*127,axis=0).astype(float).tolist()}


def quantize(actor: Actor, variant: str, calibration=None) -> QuantActor:
    actor.validate()
    if variant not in VARIANTS: raise ValueError(f'unknown variant {variant}')
    if variant == 'w8a8':
        if calibration is None: raise ValueError('W8A8 needs calibration')
        scales_in = calibration['input_scales']
        if len(scales_in) != len(actor.layers): raise ValueError('incorrect calibration size')
    layers = []
    for i, d in enumerate(actor.layers):
        w, b = np.asarray(d.weight, np.float32), np.asarray(d.bias, np.float32)
        if variant == 'fp32':
            layers.append(QuantDense(w.copy(), b.copy()))
        elif variant == 'fp16':
            layers.append(QuantDense(w.astype('<f2'), b.astype('<f2')))
        elif variant == 'bf16':
            layers.append(QuantDense(_round_bf16(w), _round_bf16(b)))
        else:
            qw, ws = _weight_quant(w)
            if variant == 'w8a8':
                xs = float(scales_in[i])
                combined = np.float64(xs) * ws.astype(np.float64)
                if not np.all(np.isfinite(combined)) or np.min(combined) <= 0:
                    raise ValueError('bad quant scale')
                qb_float = np.rint(b.astype(np.float64) / combined)

                max_acc = 127 * 127 * w.shape[0] + np.abs(qb_float)
                if not np.isfinite(max_acc).all() or np.any(max_acc >= 2**31):
                    raise ValueError(f'W8A8 INT32 accumulation/bias overflow in layer {i}')
                layers.append(QuantDense(qw, qb_float.astype('<i4'), ws, xs))
            else:
                layers.append(QuantDense(qw, b.copy(), ws))
    return QuantActor(layers, variant, actor.activation, actor.mean_limit)


def _as_float(storage, variant):
    if variant == 'bf16': 
        return _bf16_to_f32(storage)
    return np.asarray(storage).astype(np.float32)


def predict(model: QuantActor | Actor, observations: np.ndarray, *, return_mean=False):
    """Independent NumPy reference (batch or single), with explicit quant boundaries.
    W8A16 is simulated with FP16 activations, dequantized FP16 weights and FP32
    sum. FP16/BF16 variants round activation tensors at each layer boundary.
    """
    if isinstance(model, Actor): 
        model = quantize(model, 'fp32')
    x = np.asarray(observations, np.float32)
    single = x.ndim == 1
    if single: 
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != model.input_size or not np.all(np.isfinite(x)):
        raise ValueError('invalid observation dimensions/values')
    for i, layer in enumerate(model.layers):
        mode = model.variant
        if mode == 'w8a8':
            qx = _round_int8(x, layer.input_scale).astype(np.int32)
            acc = qx @ layer.weight.astype(np.int32) + layer.bias.astype(np.int32)
            x = acc.astype(np.float32) * (np.float32(layer.input_scale) * layer.weight_scales)
        elif mode in ('w8a32', 'w8a16'):
            w = layer.weight.astype(np.float32) * layer.weight_scales[None, :]
            if mode == 'w8a16':
                x = x.astype(np.float16).astype(np.float32)
                w = w.astype(np.float16).astype(np.float32)
            x = x @ w + layer.bias.astype(np.float32)
            if mode == 'w8a16': 
                x = x.astype(np.float16).astype(np.float32)
        else:
            if mode in ('fp16','bf16'):
                x = x.astype(np.float16).astype(np.float32) if mode == 'fp16' else cast_to_bf16(x)
            w = _as_float(layer.weight, mode)
            b = _as_float(layer.bias, mode)
            x = x @ w + b
            if mode == 'fp16': 
                x = x.astype(np.float16).astype(np.float32)
            if mode == 'bf16': 
                x = cast_to_bf16(x)
        if i != len(model.layers)-1:
            x = _hidden(x, model.activation).astype(np.float32)
            if mode in ('fp16', 'w8a16'): 
                x = x.astype(np.float16).astype(np.float32)
            elif mode == 'bf16': 
                x = cast_to_bf16(x)

    mean = np.float32(model.mean_limit) * np.tanh(x / np.float32(model.mean_limit))
    action = np.tanh(mean).astype(np.float32)
    if single: 
        return (action[0], mean[0]) if return_mean else action[0]

    return (action, mean) if return_mean else action
