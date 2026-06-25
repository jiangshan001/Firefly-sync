"""Core synchronisation models.

This sub-package contains the mathematical oscillator models that form
the foundation of the system:
  - Oscillator: abstract base class defining the oscillator interface.
  - KuramotoModel: continuous-phase coupled oscillator (Kuramoto dynamics).
  - PulseCoupledModel: discrete-event integrate-and-fire oscillator.
"""

from firefly_sync.core.oscillator import Oscillator
from firefly_sync.core.kuramoto import KuramotoModel
from firefly_sync.core.pulse_coupled import PulseCoupledModel

__all__ = ["Oscillator", "KuramotoModel", "PulseCoupledModel"]
