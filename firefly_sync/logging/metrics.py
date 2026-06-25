"""Synchronisation metrics for evaluating multi-oscillator coherence.

Provides quantitative measures of how synchronised a population of
oscillators is, including the classic Kuramoto order parameter and
event-based flash coincidence indices.

Step 3A adds flash-event-based synchronisation metrics for evaluating
leader-follower closed-loop trials.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from firefly_sync.simulation.drone import Drone


class SynchronizationMetrics:
    """Compute synchronisation metrics from oscillator populations.

    All methods are static; the class serves as a namespace for
    related metrics.

    Typical usage:
        r = SynchronizationMetrics.order_parameter([d.phase for d in drones])
        t_sync = SynchronizationMetrics.time_to_sync(phase_history, threshold=0.8)
    """

    @staticmethod
    def order_parameter(phases: list[float] | np.ndarray) -> float:
        """Compute the Kuramoto order parameter r.

        r = |(1/N) · Σⱼ exp(i·θⱼ)|

        where:
          - r → 0: fully incoherent (phases uniformly distributed).
          - r → 1: fully synchronised (all phases identical).

        Args:
            phases: Array of N oscillator phases in radians.

        Returns:
            Order parameter r ∈ [0, 1].
        """
        phases = np.asarray(phases, dtype=float)
        n = len(phases)
        if n == 0:
            return 0.0
        complex_sum = np.sum(np.exp(1j * phases))
        return float(np.abs(complex_sum) / n)

    @staticmethod
    def order_parameter_over_time(
        phase_history: np.ndarray,
    ) -> np.ndarray:
        """Compute the order parameter at each timestep of a simulation.

        Args:
            phase_history: Array of shape (T, N) where T is the number of
                timesteps and N is the number of oscillators.

        Returns:
            Array of shape (T,) with r(t) at each timestep.
        """
        t, n = phase_history.shape
        r_t = np.zeros(t)
        for i in range(t):
            r_t[i] = SynchronizationMetrics.order_parameter(phase_history[i])
        return r_t

    @staticmethod
    def time_to_sync(
        phase_history: np.ndarray,
        threshold: float = 0.9,
        sustain_steps: int = 50,
    ) -> float | None:
        """Determine the time (in steps) until synchronisation is achieved.

        Synchronisation is defined as the order parameter r(t) first
        crossing `threshold` and staying above it for `sustain_steps`
        consecutive timesteps (to filter transient crossings).

        Args:
            phase_history: Array of shape (T, N).
            threshold: Order parameter threshold for synchrony.
            sustain_steps: Number of consecutive steps r must stay above
                threshold before we consider sync achieved.

        Returns:
            Step index at which sync is first achieved, or None if the
            population never synchronises.
        """
        r_t = SynchronizationMetrics.order_parameter_over_time(phase_history)
        above = r_t >= threshold

        # Find first run of sustain_steps consecutive True values
        run_length = 0
        for i, val in enumerate(above):
            if val:
                run_length += 1
                if run_length >= sustain_steps:
                    return i - sustain_steps + 1
            else:
                run_length = 0
        return None

    @staticmethod
    def phase_coherence(phases: list[float] | np.ndarray) -> float:
        """Compute the mean pairwise phase coherence.

        C = (2 / (N·(N−1))) · Σ_{i<j} cos(θⱼ − θᵢ)

        Ranges from -1 (anti-phase) to +1 (perfect in-phase synchrony).

        Args:
            phases: Array of N oscillator phases.

        Returns:
            Mean phase coherence C.
        """
        phases = np.asarray(phases, dtype=float)
        n = len(phases)
        if n < 2:
            return 1.0

        coherence_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                coherence_sum += np.cos(phases[j] - phases[i])

        return float(2.0 * coherence_sum / (n * (n - 1)))

    @staticmethod
    def flash_synchrony_index(
        drones: list[Drone],
        window_steps: int = 5,
    ) -> float:
        """Compute the fraction of drones firing within a short window.

        At each timestep, we count how many drones are firing within
        `window_steps` of each other. A value of 1.0 means all drones
        always fire together.

        Note: this should be called with recorded state snapshots,
        not the live simulation state, for accurate results.

        Args:
            drones: List of drones at a single timestep.
            window_steps: Tolerance window (measured in timesteps or
                as a fraction of the period — model-dependent).

        Returns:
            Fraction of drones firing in sync (0 to 1).
        """
        # TODO: implement flash-synchrony computation from logged records
        # This will use the time_since_last_fire field from OscillatorState.
        n = len(drones)
        if n < 2:
            return 1.0

        # Simple version: fraction of drones whose last-fire time is
        # within one window of the most recent fire
        fired = [d for d in drones if d.is_firing]
        return len(fired) / n

    @staticmethod
    def frequency_dispersion(
        natural_frequencies: list[float] | np.ndarray,
    ) -> float:
        """Compute the normalised spread of natural frequencies.

        σ_ω / μ_ω — coefficient of variation of the frequency distribution.
        Lower values mean drones are more similar; higher dispersion
        makes synchronisation harder.

        Args:
            natural_frequencies: Array of natural frequencies ωᵢ.

        Returns:
            Coefficient of variation (dimensionless).
        """
        freqs = np.asarray(natural_frequencies, dtype=float)
        mean = np.mean(freqs)
        if mean == 0:
            return 0.0
        return float(np.std(freqs) / mean)

    @staticmethod
    def summary(
        phase_history: np.ndarray,
        timestep: float = 0.01,
    ) -> dict[str, float]:
        """Compute a full summary of synchronisation metrics.

        Args:
            phase_history: Array of shape (T, N).
            timestep: Simulation dt in seconds.

        Returns:
            Dictionary of metric name → value.
        """
        r_t = SynchronizationMetrics.order_parameter_over_time(phase_history)
        t_sync_steps = SynchronizationMetrics.time_to_sync(phase_history)
        time_sync = t_sync_steps * timestep if t_sync_steps is not None else None

        return {
            "final_order_parameter": float(r_t[-1]),
            "max_order_parameter": float(np.max(r_t)),
            "mean_order_parameter": float(np.mean(r_t)),
            "time_to_sync_steps": t_sync_steps,
            "time_to_sync_seconds": time_sync,
            "num_oscillators": phase_history.shape[1],
            "num_timesteps": phase_history.shape[0],
        }


# ---------------------------------------------------------------------------
# Step 3A — Flash-event-based synchronisation metrics
# ---------------------------------------------------------------------------

def pair_flash_events(
    leader_times: list[float],
    follower_times: list[float],
) -> list[dict[str, Any]]:
    """Pair each follower flash with the nearest leader flash.

    For each follower flash time, find the leader flash time closest in
    absolute value.  The signed timing error is ``leader_t − follower_t``
    so that a positive error means the leader fired *after* the follower.

    If there are no leader flashes, every timing error is ``None``.
    Pairing is one-directional (follower → nearest leader); a single
    leader flash may be paired with multiple follower flashes.

    Parameters
    ----------
    leader_times:
        Sorted list of leader flash timestamps in seconds.
    follower_times:
        Sorted list of follower flash timestamps in seconds.

    Returns
    -------
    list[dict]
        One entry per follower flash with keys ``follower_t``,
        ``leader_t``, ``timing_error_s`` (signed), ``abs_error_s``.
    """
    leader_arr = np.asarray(leader_times, dtype=float)
    pairs: list[dict[str, Any]] = []

    for ft in follower_times:
        if len(leader_arr) == 0:
            pairs.append({
                "follower_t": ft, "leader_t": None,
                "timing_error_s": None, "abs_error_s": None,
            })
            continue

        idx = int(np.argmin(np.abs(leader_arr - ft)))
        nearest_leader = float(leader_arr[idx])
        error = nearest_leader - ft
        pairs.append({
            "follower_t": round(ft, 6),
            "leader_t": round(nearest_leader, 6),
            "timing_error_s": round(error, 6),
            "abs_error_s": round(abs(error), 6),
        })

    return pairs


def check_flash_synchronisation(
    leader_times: list[float],
    follower_times: list[float],
    sync_threshold_s: float = 0.10,
    sync_cycles: int = 5,
) -> dict[str, Any]:
    """Determine whether the follower has synchronised to the leader.

    Synchronisation is declared when the absolute flash timing error
    between each follower flash and its nearest leader flash remains
    below *sync_threshold_s* for *sync_cycles* consecutive follower
    flash cycles.

    Parameters
    ----------
    leader_times:
        Sorted leader flash timestamps (seconds).
    follower_times:
        Sorted follower flash timestamps (seconds).
    sync_threshold_s:
        Maximum allowed absolute timing error (seconds).  Default 0.10.
    sync_cycles:
        Number of consecutive qualifying cycles required.  Default 5.

    Returns
    -------
    dict
        ``synchronization_success`` (bool),
        ``time_to_synchronization_s`` (float or None),
        ``sync_cycle_index`` (int or None — which follower cycle index
        first satisfies the sustained criterion).
    """
    pairs = pair_flash_events(leader_times, follower_times)

    run_length = 0
    sync_start_idx: int | None = None

    for i, p in enumerate(pairs):
        if p["abs_error_s"] is not None and p["abs_error_s"] < sync_threshold_s:
            run_length += 1
            if run_length >= sync_cycles:
                sync_start_idx = i - sync_cycles + 1
                break
        else:
            run_length = 0

    if sync_start_idx is not None and sync_start_idx < len(follower_times):
        time_to_sync = follower_times[sync_start_idx + sync_cycles - 1]
    else:
        time_to_sync = None

    return {
        "synchronization_success": sync_start_idx is not None,
        "time_to_synchronization_s": time_to_sync,
        "sync_cycle_index": sync_start_idx,
    }


def compute_flash_timing_metrics(
    leader_times: list[float],
    follower_times: list[float],
    sync_threshold_s: float = 0.10,
    sync_cycles: int = 5,
    detection_success_rate: float | None = 1.0,
    false_positive_rate: float | None = 0.0,
) -> dict[str, Any]:
    """Compute the full Step 3A metrics summary from flash event lists.

    Parameters
    ----------
    leader_times:
        Sorted leader flash timestamps.
    follower_times:
        Sorted follower flash timestamps.
    sync_threshold_s:
        Maximum absolute timing error for sync (seconds).
    sync_cycles:
        Consecutive cycles required for sync.
    detection_success_rate:
        Fraction of leader flashes detected.  In mock mode this is
        typically 1.0.  In Pi mode without ground truth it is *None*.
    false_positive_rate:
        Fraction of detected events that are false positives.  In mock
        mode this is typically 0.0.  In Pi mode without ground truth
        it is *None*.

    Returns
    -------
    dict
        The nine required Step 3A evaluation metrics.
    """
    pairs = pair_flash_events(leader_times, follower_times)
    sync_result = check_flash_synchronisation(
        leader_times, follower_times, sync_threshold_s, sync_cycles,
    )

    # Steady-state: last 50 % of follower flashes (or at least 5)
    n_steady = max(5, len(pairs) // 2)
    steady_pairs = pairs[-n_steady:]

    abs_errors = [p["abs_error_s"] for p in steady_pairs
                  if p["abs_error_s"] is not None]
    signed_errors = [p["timing_error_s"] for p in steady_pairs
                     if p["timing_error_s"] is not None]

    mean_abs = float(np.mean(abs_errors)) if abs_errors else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(abs_errors)))) if abs_errors else float("nan")
    jitter = float(np.std(abs_errors)) if len(abs_errors) >= 2 else float("nan")

    # Final frequency error: compare leader and follower mean periods
    if len(leader_times) >= 2 and len(follower_times) >= 2:
        leader_period = (leader_times[-1] - leader_times[0]) / (len(leader_times) - 1)
        follower_period = (follower_times[-1] - follower_times[0]) / (len(follower_times) - 1)
        leader_freq = 1.0 / leader_period if leader_period > 0 else 0.0
        follower_freq = 1.0 / follower_period if follower_period > 0 else 0.0
        final_freq_error = abs(leader_freq - follower_freq)
    else:
        final_freq_error = float("nan")

    # Convergence quality: 1.0 if synced and error decreasing, 0.0 if not
    if sync_result["synchronization_success"]:
        # Ratio of early-half abs error to late-half abs error
        early_half = pairs[:len(pairs) // 2]
        late_half = pairs[len(pairs) // 2:]
        early_abs = [p["abs_error_s"] for p in early_half
                     if p["abs_error_s"] is not None]
        late_abs = [p["abs_error_s"] for p in late_half
                    if p["abs_error_s"] is not None]
        early_mean = float(np.mean(early_abs)) if early_abs else float("nan")
        late_mean = float(np.mean(late_abs)) if late_abs else float("nan")
        if early_mean > 0 and late_mean > 0:
            convergence_quality = min(1.0, early_mean / late_mean)
        else:
            convergence_quality = 1.0
    else:
        convergence_quality = 0.0

    return {
        "synchronization_success": sync_result["synchronization_success"],
        "time_to_synchronization_s": sync_result["time_to_synchronization_s"],
        "steady_state_mean_abs_timing_error_s": round(mean_abs, 6),
        "steady_state_rmse_timing_error_s": round(rmse, 6),
        "steady_state_jitter_s": round(jitter, 6),
        "final_frequency_error_hz": round(final_freq_error, 6),
        "convergence_quality": round(convergence_quality, 4),
        "detection_success_rate": detection_success_rate,
        "false_positive_rate": false_positive_rate,
    }
