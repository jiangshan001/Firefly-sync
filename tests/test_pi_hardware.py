"""Hardware tests for Raspberry Pi 5.

These tests are automatically skipped on non-Pi platforms by checking
for ``/proc/device-tree/model``.  Run with::

    pytest tests/test_pi_hardware.py -v
"""

from __future__ import annotations

import sys
import pytest


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _is_raspberry_pi() -> bool:
    """Return True if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "Raspberry Pi" in f.read()
    except (FileNotFoundError, OSError):
        return False


IS_PI = _is_raspberry_pi()

skip_unless_pi = pytest.mark.skipif(
    not IS_PI,
    reason="Requires Raspberry Pi hardware (gpiozero, GPIO pins, or camera)",
)


# ---------------------------------------------------------------------------
# PiGPIOLED tests
# ---------------------------------------------------------------------------

@skip_unless_pi
class TestPiGPIOLED:
    """Integration tests for the physical LED on GPIO17.

    These tests require the LED circuit to be connected.  They blink
    briefly and verify the LED does not throw errors.
    """

    def test_construct_and_close(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        led = PiGPIOLED(pin=17)
        led.close()

    def test_on_off(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        led = PiGPIOLED(pin=17)
        try:
            led.on()
            assert led.is_flashing()
            led.off()
            assert not led.is_flashing()
        finally:
            led.close()

    def test_flash_default_duration(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        led = PiGPIOLED(pin=17, flash_duration_s=0.05)
        try:
            led.flash()
            # After flash() returns, LED should be off
            assert not led.is_flashing()
        finally:
            led.close()

    def test_flash_with_explicit_duration(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        led = PiGPIOLED(pin=17)
        try:
            led.flash(0.05)
            assert not led.is_flashing()
        finally:
            led.close()

    def test_flash_for_alias(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        led = PiGPIOLED(pin=17)
        try:
            led.flash_for(0.05)
            assert not led.is_flashing()
        finally:
            led.close()

    def test_turn_off_and_reset(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        led = PiGPIOLED(pin=17)
        try:
            led.on()
            led.turn_off()
            assert not led.is_flashing()
            led.on()
            led.reset()
            assert not led.is_flashing()
        finally:
            led.close()

    def test_context_manager(self) -> None:
        from firefly_sync.hardware.pi_led import PiGPIOLED
        with PiGPIOLED(pin=17) as led:
            led.on()
            assert led.is_flashing()


# ---------------------------------------------------------------------------
# PicameraFlashDetector tests
# ---------------------------------------------------------------------------

@skip_unless_pi
class TestPicameraFlashDetector:
    """Integration tests requiring the Pi camera (imx219 or similar)."""

    def test_start_stop(self) -> None:
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector,
        )
        det = PicameraFlashDetector(resolution=[320, 240])
        try:
            det.start()
            assert det.started
        finally:
            det.stop()
        assert not det.started

    def test_capture_frame(self) -> None:
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector,
        )
        det = PicameraFlashDetector(resolution=[320, 240])
        try:
            det.start()
            result = det.capture_frame()
            assert "brightness_mean" in result
            assert "state" in result
            assert "event_type" in result
            assert "frame_index" in result
            assert result["frame_index"] == 1
        finally:
            det.stop()

    def test_multiple_frames(self) -> None:
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector,
        )
        det = PicameraFlashDetector(resolution=[320, 240])
        try:
            det.start()
            for _ in range(5):
                result = det.capture_frame()
                assert result["state"] in ("ON", "OFF")
        finally:
            det.stop()

    def test_context_manager(self) -> None:
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector,
        )
        with PicameraFlashDetector(resolution=[320, 240]) as det:
            result = det.capture_frame()
            assert det.started
        assert not det.started
