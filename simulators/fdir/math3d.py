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
    """Returns the equivalent quaternion whose scalar part is non-negative."""
    return jnp.where(q[..., :1] < 0.0, -q, q)


def attitude_error(target_q: jax.Array, current_q: jax.Array) -> jax.Array:
    """Shortest error rotation from target attitude to current attitude."""
    return canonicalize_quat(
        quat_multiply(quat_conjugate(target_q), current_q)
    )


def attitude_angle(error_q: jax.Array) -> jax.Array:
    """Geodesic attitude error in radians, in [0, pi]."""
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


def rotation_vector_to_quat(rotation_vector: jax.Array) -> jax.Array:
    """Converts body-frame rotation vectors [rad] to unit quaternions."""
    angle = jnp.linalg.norm(rotation_vector, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = jnp.where(
        angle > 1.0e-7,
        jnp.sin(half_angle) / angle,
        0.5 - jnp.square(angle) / 48.0,
    )
    return quat_normalize(
        jnp.concatenate(
            [jnp.cos(half_angle), rotation_vector * scale], axis=-1
        )
    )



def quat_to_rotation_vector(q: jax.Array) -> jax.Array:
    """Returns the shortest rotation vector represented by ``q``."""
    q = canonicalize_quat(quat_normalize(q))
    vector = q[..., 1:]
    vector_norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * jnp.arctan2(vector_norm, jnp.clip(q[..., :1], 0.0, 1.0))
    scale = jnp.where(
        vector_norm > 1.0e-8,
        angle / vector_norm,
        2.0 + jnp.square(vector_norm) / 3.0,
    )
    return vector * scale


def rotation_matrix_to_quat(matrix: jax.Array) -> jax.Array:
    """Batch-safe conversion from proper 3x3 rotation matrices to quaternions."""
    r00 = matrix[..., 0, 0]
    r11 = matrix[..., 1, 1]
    r22 = matrix[..., 2, 2]
    w = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1.0 + r00 + r11 + r22))
    x = 0.5 * jnp.sign(matrix[..., 2, 1] - matrix[..., 1, 2] + 1.0e-12) * jnp.sqrt(
        jnp.maximum(0.0, 1.0 + r00 - r11 - r22)
    )
    y = 0.5 * jnp.sign(matrix[..., 0, 2] - matrix[..., 2, 0] + 1.0e-12) * jnp.sqrt(
        jnp.maximum(0.0, 1.0 - r00 + r11 - r22)
    )
    z = 0.5 * jnp.sign(matrix[..., 1, 0] - matrix[..., 0, 1] + 1.0e-12) * jnp.sqrt(
        jnp.maximum(0.0, 1.0 - r00 - r11 + r22)
    )
    return canonicalize_quat(quat_normalize(jnp.stack([w, x, y, z], axis=-1)))

def integrate_quaternion(q: jax.Array, omega_body: jax.Array, dt: float) -> jax.Array:
    """Exponential-map integration for constant body rate over ``dt``."""
    delta_q = rotation_vector_to_quat(omega_body * dt)
    return quat_normalize(quat_multiply(q, delta_q))




def rotate_body_to_inertial(q: jax.Array, vector_body: jax.Array) -> jax.Array:
    """Rotates body-frame vectors to inertial coordinates using ``q``."""
    q = quat_normalize(q)
    q_vec = q[..., 1:]
    twice_cross = 2.0 * jnp.cross(q_vec, vector_body)
    return (
        vector_body
        + q[..., :1] * twice_cross
        + jnp.cross(q_vec, twice_cross)
    )


def rotate_inertial_to_body(q: jax.Array, vector_inertial: jax.Array) -> jax.Array:
    """Rotates inertial-frame vectors to body coordinates using ``q``."""
    return rotate_body_to_inertial(quat_conjugate(q), vector_inertial)


def sample_quaternion_in_cone(key: jax.Array, batch_size: int, max_angle_rad: float) -> jax.Array:
    axis_key, angle_key = jax.random.split(key)
    axes = jax.random.normal(axis_key, shape=(batch_size, 3))
    axes = axes / (jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1.0e-12)
    angles = jax.random.uniform(
        angle_key, shape=(batch_size,), minval=0.0, maxval=max_angle_rad
    )
    return axis_angle_to_quat(axes, angles)
