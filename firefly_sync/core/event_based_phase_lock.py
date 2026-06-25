"""Event-Based Adaptive Phase/Frequency Locking (EAPF) — Model 3.

Engineering-oriented synchronisation model inspired by phase-locked loop
(PLL) techniques (Gardner 2005).  On each detected leader flash, the
follower measures the phase error and applies independent phase and
frequency corrections.

Pure core model — no camera, no GPIO, no API.  Unit-testable on any
platform.

See: ``docs/synchronisation_models/model_3_event_based_phase_frequency_locking.md``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle in radians to [−π, π)."""
    return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi


def wrap_to_2pi(angle_rad: float) -> float:
    """Wrap an angle in radians to [0, 2π)."""
    return angle_rad % (2.0 * math.pi)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EventBasedPhaseLockConfig:
    """Configuration for the EAPF oscillator.

    Attributes:
        natural_frequency_hz: Initial follower frequency in Hz.
        phase_gain: Gain for phase correction per leader event.
        frequency_gain: Gain for frequency correction per leader event.
        frequency_min_hz: Minimum allowed follower frequency.
        frequency_max_hz: Maximum allowed follower frequency.
        leader_period_window: Number of recent intervals for median
            period estimate.
    """

    natural_frequency_hz: float = 1.5
    phase_gain: float = 0.20
    frequency_gain: float = 0.05
    frequency_min_hz: float = 0.5
    frequency_max_hz: float = 4.0
    leader_period_window: int = 6


# ---------------------------------------------------------------------------
# Oscillator
# ---------------------------------------------------------------------------

class EventBasedPhaseLockOscillator:
    """Event-based adaptive phase/frequency locking oscillator (EAPF).

    Loop order (every step):
        1. Compute measured dt.
        2. Advance natural phase: ``phase_rad += omega * dt``.
        3. If phase wraps past 2π → follower flash, phase −= 2π.
        4. If leader flash event:
           a. Estimate leader period (median of recent intervals).
           b. Compute signed phase error: ``wrap_to_pi(0 − phase_rad)``.
           c. Phase correction: ``phase_rad += phase_gain * error``.
           d. Frequency correction: ``f += freq_gain * error/(2π)``, clamped.
           e. Update omega.
        5. Return state dictionary.

    Design choice: follower flash is triggered by natural phase wrap
    (step 3), NOT by correction-induced wrap.  This keeps the follower
    output timing predictable and avoids double-firing during correction.
    """

    def __init__(self, config: EventBasedPhaseLockConfig | None = None) -> None:
        self.config = config or EventBasedPhaseLockConfig()
        self._phase_rad: float = 0.0
        self._frequency_hz: float = self.config.natural_frequency_hz
        self._omega_rad_s: float = 2.0 * math.pi * self._frequency_hz
        self._leader_flash_times: list[float] = []
        self._leader_period_estimate_s: float = (
            1.0 / self.config.natural_frequency_hz
            if self.config.natural_frequency_hz > 0
            else 0.5
        )
        self._fire_count: int = 0
        self._last_flash_time_s: float | None = None

    # -- properties --

    @property
    def phase_rad(self) -> float:
        return self._phase_rad

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def omega_rad_s(self) -> float:
        return self._omega_rad_s

    @property
    def leader_period_estimate_s(self) -> float:
        return self._leader_period_estimate_s

    @property
    def fire_count(self) -> int:
        return self._fire_count

    @property
    def leader_flash_count(self) -> int:
        return len(self._leader_flash_times)

    # -- core step --

    def step(
        self,
        dt_s: float,
        leader_flash_event: bool,
        t_s: float,
    ) -> dict[str, Any]:
        """Advance the oscillator by one timestep.

        Parameters
        ----------
        dt_s:
            Measured wall-clock time step in seconds.
        leader_flash_event:
            True if a valid leader flash rising edge was detected.
        t_s:
            Current monotonic time in seconds.

        Returns
        -------
        dict
            Keys: ``phase_rad``, ``frequency_hz``, ``omega_rad_s``,
            ``phase_error_rad``, ``leader_period_estimate_s``,
            ``follower_flash_event``, ``leader_flash_event_used``,
            ``fire_count``.
        """
        leader_used = False
        phase_error_rad = 0.0

        # Step 2: advance natural phase
        if dt_s > 0:
            self._phase_rad += self._omega_rad_s * dt_s

        # Step 3: natural phase wrap → follower flash
        follower_flash = False
        if self._phase_rad >= 2.0 * math.pi:
            follower_flash = True
            self._fire_count += 1
            self._last_flash_time_s = t_s
            self._phase_rad -= 2.0 * math.pi

        # Step 4: leader event processing
        if leader_flash_event:
            leader_used = True
            self._leader_flash_times.append(t_s)

            # 4a. Estimate leader period (median of recent intervals)
            self._update_period_estimate()

            # 4b. Compute signed phase error
            # Desired follower phase at leader flash = 0 rad
            # phase_error > 0 → follower is lagging → must catch up
            # phase_error < 0 → follower is leading → must slow down
            phase_error_rad = wrap_to_pi(0.0 - self._phase_rad)

            # 4c. Phase correction
            self._phase_rad += self.config.phase_gain * phase_error_rad
            self._phase_rad = wrap_to_2pi(self._phase_rad)

            # 4d. Frequency correction
            freq_correction = (self.config.frequency_gain
                               * phase_error_rad / (2.0 * math.pi))
            self._frequency_hz = clamp(
                self._frequency_hz + freq_correction,
                self.config.frequency_min_hz,
                self.config.frequency_max_hz,
            )

            # 4e. Update omega
            self._omega_rad_s = 2.0 * math.pi * self._frequency_hz

        return {
            "phase_rad": round(self._phase_rad, 6),
            "frequency_hz": round(self._frequency_hz, 6),
            "omega_rad_s": round(self._omega_rad_s, 6),
            "phase_error_rad": round(phase_error_rad, 6),
            "leader_period_estimate_s": round(self._leader_period_estimate_s, 6),
            "leader_flash_count": len(self._leader_flash_times),
            "follower_flash_event": follower_flash,
            "leader_flash_event_used": leader_used,
            "fire_count": self._fire_count,
        }

    # -- lifecycle --

    def reset(self) -> None:
        """Reset oscillator to initial state."""
        self._phase_rad = 0.0
        self._frequency_hz = self.config.natural_frequency_hz
        self._omega_rad_s = 2.0 * math.pi * self._frequency_hz
        self._leader_flash_times.clear()
        self._leader_period_estimate_s = (
            1.0 / self.config.natural_frequency_hz
            if self.config.natural_frequency_hz > 0
            else 0.5
        )
        self._fire_count = 0
        self._last_flash_time_s = None

    # -- internal --

    def _update_period_estimate(self) -> None:
        """Update leader period from recent flash intervals (median)."""
        if len(self._leader_flash_times) < 2:
            return
        w = self.config.leader_period_window
        recent = self._leader_flash_times[-w:]
        if len(recent) < 2:
            return
        intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        intervals.sort()
        n = len(intervals)
        if n % 2 == 1:
            self._leader_period_estimate_s = intervals[n // 2]
        else:
            self._leader_period_estimate_s = (intervals[n // 2 - 1]
                                              + intervals[n // 2]) / 2.0
