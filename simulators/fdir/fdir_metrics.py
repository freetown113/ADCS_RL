from typing import NamedTuple

import jax
import jax.numpy as jnp

from simulators.fdir.fdir import DEGRADED, FAILED


class BinaryDetectionMetrics(NamedTuple):
    false_alarm_rate: jax.Array
    missed_detection_rate: jax.Array
    mean_detection_latency_s: jax.Array
    detection_count: jax.Array


class WheelFDIRMetrics(NamedTuple):
    detection: BinaryDetectionMetrics
    isolation_accuracy: jax.Array
    authority_mae: jax.Array


def _first_true_index(mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns first true index along time axis and whether one exists."""
    exists = jnp.any(mask, axis=0)
    index = jnp.argmax(mask.astype(jnp.int32), axis=0)
    return index, exists


def binary_detection_metrics(
    truth_fault: jax.Array,
    detected_fault: jax.Array,
    dt_s: float,
) -> BinaryDetectionMetrics:
    """Metrics for boolean histories shaped [time, ...]."""
    truth_fault = truth_fault.astype(jnp.bool_)
    detected_fault = detected_fault.astype(jnp.bool_)
    healthy = ~truth_fault
    false_alarm_rate = jnp.sum(detected_fault & healthy) / jnp.maximum(jnp.sum(healthy), 1)

    truth_start, truth_exists = _first_true_index(truth_fault)
    detect_start, detect_exists = _first_true_index(detected_fault & truth_fault)
    missed = truth_exists & (~detect_exists)
    missed_detection_rate = jnp.sum(missed) / jnp.maximum(jnp.sum(truth_exists), 1)

    latency_steps = jnp.maximum(detect_start - truth_start, 0)
    valid_latency = truth_exists & detect_exists
    mean_latency = (
        jnp.sum(jnp.where(valid_latency, latency_steps, 0))
        / jnp.maximum(jnp.sum(valid_latency), 1)
        * dt_s
    )
    return BinaryDetectionMetrics(
        false_alarm_rate=false_alarm_rate,
        missed_detection_rate=missed_detection_rate,
        mean_detection_latency_s=mean_latency,
        detection_count=jnp.sum(valid_latency),
    )


def wheel_fdir_metrics(
    truth_authority: jax.Array,
    estimated_authority: jax.Array,
    wheel_health: jax.Array,
    dt_s: float,
    truth_fault_threshold: float = 0.95,
) -> WheelFDIRMetrics:
    """Evaluate wheel FDIR histories shaped [time,batch,4]."""
    truth_fault = truth_authority < truth_fault_threshold
    detected_fault = (wheel_health == DEGRADED) | (wheel_health == FAILED)
    detection = binary_detection_metrics(truth_fault, detected_fault, dt_s)

    any_truth = jnp.any(truth_fault, axis=-1)
    any_detected = jnp.any(detected_fault, axis=-1)
    truth_id = jnp.argmin(truth_authority, axis=-1)
    detected_id = jnp.argmin(estimated_authority, axis=-1)
    comparable = any_truth & any_detected
    isolation_accuracy = jnp.sum(comparable & (truth_id == detected_id)) / jnp.maximum(jnp.sum(comparable), 1)
    authority_mae = jnp.mean(jnp.abs(truth_authority - estimated_authority))
    return WheelFDIRMetrics(detection, isolation_accuracy, authority_mae)
