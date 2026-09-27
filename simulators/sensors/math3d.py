import jax
import jax.numpy as jnp


def quat_conjugate(q: jax.Array) -> jax.Array:
    signs = jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)
    return q * signs


def quat_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = [q1[..., i] for i in range(4)]
    w2, x2, y2, z2 = [q2[..., i] for i in range(4)]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_normalize(q: jax.Array, eps: float = 1.0e-12) -> jax.Array:
    norm = jnp.sqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + eps)
    return q / norm


def canonicalize_quat(q: jax.Array) -> jax.Array:
    return jnp.where(q[..., :1] < 0.0, -q, q)


def attitude_error(target_q: jax.Array, current_q: jax.Array) -> jax.Array:
    return canonicalize_quat(
        quat_multiply(quat_conjugate(target_q), current_q)
    )


def attitude_angle(error_q: jax.Array) -> jax.Array:
    q = canonicalize_quat(quat_normalize(error_q))
    vector_norm = jnp.linalg.norm(q[..., 1:], axis=-1)
    scalar = jnp.clip(q[..., 0], 0.0, 1.0)
    return 2.0 * jnp.arctan2(vector_norm, scalar)


def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
    axis = axis / (jnp.linalg.norm(axis, axis=-1, keepdims=True) + 1.0e-12)
    half = 0.5 * angle
    return quat_normalize(
        jnp.concatenate(
            [jnp.cos(half)[..., None], axis * jnp.sin(half)[..., None]],
            axis=-1,
        )
    )


def integrate_quaternion(q: jax.Array, omega_body: jax.Array, dt: float) -> jax.Array:
    rotation_vector = omega_body * dt
    angle = jnp.linalg.norm(rotation_vector, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = jnp.where(
        angle > 1.0e-7,
        jnp.sin(half_angle) / angle,
        0.5 - jnp.square(angle) / 48.0,
    )
    delta_q = jnp.concatenate(
        [jnp.cos(half_angle), rotation_vector * scale], axis=-1
    )
    return quat_normalize(quat_multiply(q, delta_q))




def rotate_body_to_inertial(q: jax.Array, vector_body: jax.Array) -> jax.Array:
    q = quat_normalize(q)
    q_vec = q[..., 1:]
    twice_cross = 2.0 * jnp.cross(q_vec, vector_body)
    return (
        vector_body
        + q[..., :1] * twice_cross
        + jnp.cross(q_vec, twice_cross)
    )


def rotate_inertial_to_body(q: jax.Array, vector_inertial: jax.Array) -> jax.Array:
    return rotate_body_to_inertial(quat_conjugate(q), vector_inertial)


def sample_quaternion_in_cone(key: jax.Array, batch_size: int, max_angle_rad: float) -> jax.Array:
    axis_key, angle_key = jax.random.split(key)
    axes = jax.random.normal(axis_key, shape=(batch_size, 3))
    axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1.0e-12)
    angles = jax.random.uniform(
        angle_key, shape=(batch_size,), minval=0.0, maxval=max_angle_rad
    )
    return axis_angle_to_quat(axes, angles)
