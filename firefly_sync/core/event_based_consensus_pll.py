"""Event-Based Consensus Phase-Locked Loop — Multi-Neighbour Variant.

Inspired by distributed phase/frequency agreement (Olfati-Saber et al. 2007)
and PLL engineering (Gardner 2005).  Each agent maintains neighbour state
estimates and applies phase + frequency consensus corrections.

Unlike the fixed-leader EAPF tracker, this variant is designed for mutual
multi-neighbour synchronisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _wrap_pi(a: float) -> float:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def _wrap_2pi(a: float) -> float:
    return a % (2 * math.pi)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class ConsensusPLLConfig:
    natural_frequency_hz: float = 2.0
    phase_gain: float = 0.05
    frequency_gain: float = 0.02
    phase_error_filter_alpha: float = 0.2
    frequency_error_filter_alpha: float = 0.2
    max_phase_step_rad: float = 0.2
    max_frequency_step_hz: float = 0.05
    frequency_min_hz: float = 0.5
    frequency_max_hz: float = 4.0
    allow_correction_induced_flash: bool = False
    neighbour_period_window: int = 6


class EventBasedConsensusPLLOscillator:
    """Consensus PLL oscillator for multi-neighbour synchronisation."""

    def __init__(self, config: ConsensusPLLConfig | None = None) -> None:
        self.config = config or ConsensusPLLConfig()
        self._phase_rad: float = 0.0
        self._frequency_hz: float = self.config.natural_frequency_hz
        self._omega_rad_s: float = 2.0 * math.pi * self._frequency_hz
        self._fire_count: int = 0
        self._last_flash_time_s: float | None = None
        # Neighbour state
        self._neighbour_last_t: dict[int, float] = {}
        self._neighbour_periods: dict[int, list[float]] = {}
        self._neighbour_period_est: dict[int, float] = {}
        self._neighbour_freq_est: dict[int, float] = {}
        self._neighbour_phase_est: dict[int, float] = {}
        # Filtered errors
        self._phase_error_filt: float = 0.0
        self._freq_error_filt: float = 0.0

    @property
    def phase_rad(self) -> float:
        return self._phase_rad

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def fire_count(self) -> int:
        return self._fire_count

    def record_neighbour_flash(self, neighbour_id: int, t_s: float) -> None:
        """Update neighbour state estimate on observed flash."""
        if neighbour_id in self._neighbour_last_t:
            interval = t_s - self._neighbour_last_t[neighbour_id]
            if interval > 0:
                self._neighbour_periods.setdefault(neighbour_id, []).append(interval)
                w = self.config.neighbour_period_window
                if len(self._neighbour_periods[neighbour_id]) > w * 2:
                    self._neighbour_periods[neighbour_id].pop(0)
                recent = self._neighbour_periods[neighbour_id][-w:]
                if recent:
                    p = float(np.median(recent))
                    self._neighbour_period_est[neighbour_id] = p
                    self._neighbour_freq_est[neighbour_id] = 1.0 / p if p > 0 else 0.0
        self._neighbour_last_t[neighbour_id] = t_s
        self._neighbour_phase_est[neighbour_id] = 0.0  # reset at flash

    def propagate_neighbour_phases(self, dt_s: float) -> None:
        """Advance neighbour phase estimates between flashes."""
        for nid in list(self._neighbour_phase_est.keys()):
            freq = self._neighbour_freq_est.get(nid, self._frequency_hz)
            self._neighbour_phase_est[nid] = (
                self._neighbour_phase_est[nid] + 2.0 * math.pi * freq * dt_s
            ) % (2.0 * math.pi)

    def has_neighbour(self, neighbour_id: int) -> bool:
        return neighbour_id in self._neighbour_phase_est

    def step(self, dt_s: float, t_s: float,
             neighbour_flash_ids: list[int] | None = None) -> dict[str, Any]:
        """Advance one step.  neighbour_flash_ids lists which neighbours fired."""
        if neighbour_flash_ids is None:
            neighbour_flash_ids = []

        # Record neighbour flashes
        for nid in neighbour_flash_ids:
            self.record_neighbour_flash(nid, t_s)

        # 1. Advance own phase
        if dt_s > 0:
            self._phase_rad += self._omega_rad_s * dt_s

        # 2. Natural wrap → flash
        follower_flash = False
        if self._phase_rad >= 2.0 * math.pi:
            if not self.config.allow_correction_induced_flash:
                follower_flash = True
                self._fire_count += 1
                self._last_flash_time_s = t_s
                self._phase_rad -= 2.0 * math.pi

        # 3. Propagate neighbour phases
        self.propagate_neighbour_phases(dt_s)

        # 4. Consensus correction (if neighbours exist)
        phase_error = 0.0
        freq_error = 0.0
        visible = [n for n in self._neighbour_phase_est.keys()
                   if n not in neighbour_flash_ids or True]  # all known neighbours
        if visible:
            p_errors = [_wrap_pi(self._neighbour_phase_est[n] - self._phase_rad)
                        for n in visible]
            f_errors = [self._neighbour_freq_est.get(n, self._frequency_hz) - self._frequency_hz
                        for n in visible]
            phase_error = float(np.mean(p_errors)) if p_errors else 0.0
            freq_error = float(np.mean(f_errors)) if f_errors else 0.0

        # Low-pass filter errors
        alpha_p = self.config.phase_error_filter_alpha
        alpha_f = self.config.frequency_error_filter_alpha
        self._phase_error_filt = alpha_p * phase_error + (1 - alpha_p) * self._phase_error_filt
        self._freq_error_filt = alpha_f * freq_error + (1 - alpha_f) * self._freq_error_filt

        # Apply phase correction (rate-limited)
        phase_step = _clamp(self.config.phase_gain * self._phase_error_filt,
                            -self.config.max_phase_step_rad,
                            self.config.max_phase_step_rad)
        self._phase_rad = _wrap_2pi(self._phase_rad + phase_step)

        # Apply frequency correction (rate-limited)
        freq_step = _clamp(self.config.frequency_gain * self._freq_error_filt,
                           -self.config.max_frequency_step_hz,
                           self.config.max_frequency_step_hz)
        self._frequency_hz = _clamp(self._frequency_hz + freq_step,
                                    self.config.frequency_min_hz,
                                    self.config.frequency_max_hz)
        self._omega_rad_s = 2.0 * math.pi * self._frequency_hz

        # Correction-induced flash
        if self.config.allow_correction_induced_flash and self._phase_rad < 0.01:
            if self._last_flash_time_s is None or (t_s - self._last_flash_time_s) > 0.05:
                follower_flash = True
                self._fire_count += 1
                self._last_flash_time_s = t_s
                self._phase_rad = 0.0

        return {
            "phase_rad": round(self._phase_rad, 6),
            "frequency_hz": round(self._frequency_hz, 6),
            "omega_rad_s": round(self._omega_rad_s, 6),
            "phase_error_rad": round(phase_error, 6),
            "freq_error_hz": round(freq_error, 6),
            "follower_flash_event": follower_flash,
            "fire_count": self._fire_count,
            "n_neighbours": len(visible),
        }

    def reset(self) -> None:
        self._phase_rad = 0.0
        self._frequency_hz = self.config.natural_frequency_hz
        self._omega_rad_s = 2.0 * math.pi * self._frequency_hz
        self._fire_count = 0
        self._last_flash_time_s = None
        self._neighbour_last_t.clear()
        self._neighbour_periods.clear()
        self._neighbour_period_est.clear()
        self._neighbour_freq_est.clear()
        self._neighbour_phase_est.clear()
        self._phase_error_filt = 0.0
        self._freq_error_filt = 0.0
