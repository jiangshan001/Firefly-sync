"""Hardware abstraction layer.

Provides abstract interfaces and mock implementations for:
  - LED output (physical GPIO or simulated console output).
  - Camera / flash detection (physical camera or simulated neighbour query).
  - Flight controller (MAVLink or no-op stub).
  - Flash detection state machine (pure Python, no hardware deps).

All real-hardware implementations should subclass the abstract bases
defined here, ensuring a clean swap between simulation and hardware modes.
"""

from firefly_sync.hardware.led import AbstractLED, MockLED
from firefly_sync.hardware.camera import AbstractCamera, MockCamera
from firefly_sync.hardware.flight_controller import (
    AbstractFlightController,
    MockFlightController,
)

# Pure-Python detection logic (no hardware deps — always importable).
from firefly_sync.hardware.flash_detector import (
    FlashDetectorState,
    compute_bright_blob_metrics,
    compute_local_contrast,
    compute_roi_mean_brightness,
    compute_roi_median_brightness,
    compute_top_percentile_brightness,
    estimate_frequency,
    update_flash_detector,
)

# Adaptive signal detection (pure Python + numpy, no Pi hardware).
from firefly_sync.hardware.signal_detector import (
    AdaptiveDetectorState,
    RollingAdaptiveSignalDetector,
    estimate_frequency_autocorrelation,
)

# ROI localisation (pure numpy + cv2, no Pi hardware).
from firefly_sync.hardware.roi_locator import locate_flashing_region

# Pi hardware modules.  The modules themselves can be imported on any
# platform, but constructing instances will raise a clear ImportError
# if the required Pi libraries (gpiozero, picamera2) are missing.
from firefly_sync.hardware.pi_led import PiGPIOLED
from firefly_sync.hardware.picamera_flash_detector import PicameraFlashDetector

__all__ = [
    # Abstract / mock
    "AbstractLED",
    "MockLED",
    "AbstractCamera",
    "MockCamera",
    "AbstractFlightController",
    "MockFlightController",
    # Pi hardware
    "PiGPIOLED",
    "PicameraFlashDetector",
    # Pure detection logic
    "FlashDetectorState",
    "update_flash_detector",
    "estimate_frequency",
    "compute_roi_mean_brightness",
    "compute_roi_median_brightness",
    "compute_top_percentile_brightness",
    "compute_bright_blob_metrics",
    "compute_local_contrast",
    # Adaptive signal detection
    "RollingAdaptiveSignalDetector",
    "AdaptiveDetectorState",
    "estimate_frequency_autocorrelation",
    # ROI localisation
    "locate_flashing_region",
]
