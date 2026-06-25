"""Raspberry Pi GPIO LED wrapper — extends AbstractLED.

Provides a real-hardware LED implementation using ``gpiozero.LED``.
All hardware imports are lazy — the module can be imported on any
platform, but instantiating ``PiGPIOLED`` requires ``gpiozero`` and a
Raspberry Pi.

Usage::

    from firefly_sync.hardware.pi_led import PiGPIOLED

    led = PiGPIOLED(pin=17)
    led.on()
    led.off()
    led.flash()              # AbstractLED — uses default flash duration
    led.flash(0.3)           # alias for flash_for() — Pi extension
    led.close()
"""

from __future__ import annotations

import time
import warnings

from firefly_sync.hardware.led import AbstractLED

# Delayed imports — populated on first construction.
_LED_CLASS = None
_GPIOZERO_MSG = (
    "gpiozero is required for PiGPIOLED. "
    "On Raspberry Pi OS:  sudo apt install -y python3-gpiozero\n"
    "Or via pip:          pip install gpiozero"
)


def _ensure_gpiozero():
    """Lazy-import gpiozero.LED at construction time.

    Raises
    ------
    ImportError
        If gpiozero is not installed or the platform is not a Raspberry Pi.
    """
    global _LED_CLASS
    if _LED_CLASS is not None:
        return
    try:
        from gpiozero import LED  # type: ignore[import-untyped]
        _LED_CLASS = LED
    except ImportError:
        raise ImportError(_GPIOZERO_MSG)


# ---------------------------------------------------------------------------
# PiGPIOLED
# ---------------------------------------------------------------------------

class PiGPIOLED(AbstractLED):
    """Real GPIO LED on a Raspberry Pi.

    Implements the ``AbstractLED`` interface and adds Pi-specific
    ``on()`` / ``off()`` / ``flash(duration_s)`` / ``close()`` methods.

    Parameters
    ----------
    pin:
        BCM GPIO pin number (default 17).
    flash_duration_s:
        Default flash duration in seconds used by the no-argument
        ``flash()`` method inherited from ``AbstractLED``.
    """

    def __init__(self, pin: int = 17, flash_duration_s: float = 0.1) -> None:
        _ensure_gpiozero()
        assert _LED_CLASS is not None  # guarded by _ensure_gpiozero
        self._pin = pin
        self._flash_duration_s = float(flash_duration_s)
        self._led = _LED_CLASS(pin)
        self._led.off()  # ensure LED starts in known state

    # ---- AbstractLED implementation ----

    def flash(self, duration_s: float | None = None) -> None:
        """Emit a single flash.

        Turns the LED on for *duration_s* seconds then off.  If
        *duration_s* is omitted the constructor default is used.

        This method satisfies the ``AbstractLED.flash()`` no-argument
        contract while also accepting an optional duration (the extra
        parameter is ignored by callers that use the abstract interface).
        """
        dur = duration_s if duration_s is not None else self._flash_duration_s
        self._led.on()
        # Busy-wait is fine for short flashes (< 200 ms).
        # For longer durations a threaded approach could be used,
        # but hardware-in-the-loop experiments use brief pulses.
        target = time.perf_counter() + dur
        while time.perf_counter() < target:
            pass
        self._led.off()

    def flash_for(self, duration_s: float) -> None:
        """PiGPIOLED-specific: flash for an explicit duration.

        Alias for ``flash(duration_s)`` provided for readability.
        """
        self.flash(duration_s=duration_s)

    def is_flashing(self) -> bool:
        """Check whether the LED is currently lit.

        Returns
        -------
        bool
            ``True`` if the LED is actively emitting light.
        """
        return bool(self._led.is_lit)

    def turn_off(self) -> None:
        """Force the LED off immediately."""
        self._led.off()

    def reset(self) -> None:
        """Reset LED state — equivalent to ``turn_off()``."""
        self._led.off()

    # ---- PiGPIOLED-specific convenience methods ----

    def on(self) -> None:
        """Turn the LED on (stays on until ``off()`` is called)."""
        self._led.on()

    def off(self) -> None:
        """Turn the LED off."""
        self._led.off()

    def close(self) -> None:
        """Release the GPIO resource.  Call once when finished."""
        try:
            self._led.off()
            self._led.close()
        except Exception:
            pass

    # ---- Context manager support ----

    def __enter__(self) -> PiGPIOLED:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
