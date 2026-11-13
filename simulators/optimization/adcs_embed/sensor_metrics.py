import numpy as np

SENSORS = ('gyro','wheel','star_tracker','magnetometer','sun_sensor','gnss')


def _longest_false_runs(valid: np.ndarray) -> tuple[int,int]:
    """Return (maximum invalid steps, maximum invalid steps *after first valid*).
    Aggregate across individual episode/channel columns; time is axis 0.
    """
    v = valid.reshape(valid.shape[0], -1)
    max_any = max_post = 0
    for j in range(v.shape[1]):
        column = v[:,j]
        length = 0
        ever_valid = False
        for value in column:
            if value:
                length = 0
                ever_valid = True
            else:
                length += 1
                max_any = max(max_any, length)
                if ever_valid:
                    max_post = max(max_post, length)
    return max_any,max_post


def summarize_sensor(ages, valid, *, dt: float) -> dict:
    a=np.asarray(ages,dtype=np.float64)
    v=np.asarray(valid,dtype=bool)
    if a.shape != v.shape or a.ndim < 2:
        raise ValueError('age and validity must have identical [T, B, ...] shapes')
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('control timestep dt must be positive finite')
    total = int(v.size)
    good = v & np.isfinite(a)
    finite_a = a[good]
    inf = np.isinf(a)
    invalid = ~v
    max_any,max_post = _longest_false_runs(v)
    v_flat = v.reshape(v.shape[0], -1)
    first_valid = np.argmax(v_flat,axis=0)
    has_valid = np.any(v_flat,axis=0)
    first_valid_seconds = np.where(has_valid, first_valid * dt, np.nan)

    losses = v[:-1] & ~v[1:]
    acquisitions = ~v[:-1] & v[1:]
    invalid_indices=np.flatnonzero(invalid.ravel())[:12]
    examples=[]

    for flat in invalid_indices:
        position=tuple(int(i) for i in np.unravel_index(flat,v.shape))
        examples.append({'step':position[0], 'episode':position[1],
                         'extra_channel_indices':list(position[2:]),
                         'elapsed_seconds':float((position[0]+1)*dt)})
        
    def f_or_none(vals,fn):
        return float(fn(vals)) if vals.size else None
    
    return {
        'sample_count':total,
        'valid_fraction':float(np.mean(v)),
        'invalid_sample_count':int(np.sum(invalid)),
        'infinite_age_count':int(np.sum(inf)),
        'infinite_age_while_invalid_count':int(np.sum(inf & invalid)),
        'infinite_age_while_valid_count':int(np.sum(inf & v)),
        'nonfinite_age_while_valid_count':int(np.sum(~np.isfinite(a) & v)),
        'finite_valid_age_max_seconds':f_or_none(finite_a,np.max),
        'finite_valid_age_p99_seconds':f_or_none(finite_a,lambda u:np.percentile(u,99)),
        'finite_valid_age_mean_seconds':f_or_none(finite_a,np.mean),
        'longest_invalid_seconds':float(max_any * dt),
        'longest_invalid_after_first_valid_seconds':float(max_post * dt),
        'fraction_never_valid':float(np.mean(~has_valid)),
        'first_valid_time_mean_seconds':f_or_none(first_valid_seconds[np.isfinite(first_valid_seconds)],np.mean),
        'invalid_at_first_step_fraction':float(np.mean(~v_flat[0])),
        'invalid_at_final_step_fraction':float(np.mean(~v_flat[-1])),
        'loss_events_valid_to_invalid':int(np.count_nonzero(losses)),
        'acquisition_events_invalid_to_valid':int(np.count_nonzero(acquisitions)),
        'invalid_sample_examples':examples,
    }


def trajectory_sensor_report(trajectory, *, dt: float) -> dict:
    """Accept ADCS EvaluationTrajectory or equivalent attribute object.
    Move JAX trajectory to host before invoking if running under device memory
    pressure. This function is not meant to be used inside jax.jit.
    """
    results={}
    for sensor in SENSORS:
        age=getattr(trajectory,f'{sensor}_age_seconds')
        valid=getattr(trajectory,f'{sensor}_valid')
        results[sensor]=summarize_sensor(age,valid,dt=dt)
    return results


def rms_pointing_error_deg(angle_rad) -> float | None:
    """Full-trajectory RMS, all time steps and episodes, in degrees.
    Returns None rather than contaminating a JSON summary with NaN or Infinity.
    This matches the original evaluator's `pointing_rms_deg` definition.
    """
    a=np.asarray(angle_rad,dtype=np.float64)
    if a.size==0 or not np.isfinite(a).all():
        return None
    return float(np.rad2deg(np.sqrt(np.mean(a*a))))
