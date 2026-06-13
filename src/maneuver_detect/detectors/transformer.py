"""The transformer learned detector — localise gaps with the model, invert the physics for Δv/type.

``TransformerDetector`` is the inference side of the second learned baseline. Like the BiLSTM it
inherits the whole architecture-agnostic pipeline from
:class:`~maneuver_detect.detectors.learned._LearnedDetector` — load a checkpoint bundle, run the
bare network over the V5-encoded series on **CPU**, and turn the per-token maneuver probabilities
into the canonical schema via the same Gauss inversion the classical detector uses. The architecture
is selected from the bundle's ``network`` tag at load time, so the shared base rebuilds the
transformer here and the BiLSTM for its sibling detector.

The detector registers under ``"transformer-base"``. It needs a trained checkpoint: construct it
with a bundle (or a path), or set the :data:`CHECKPOINT_ENV` environment variable to a bundle path
so the no-argument construction the registry uses (``detect(history, model="transformer-base")``)
finds one. Without a checkpoint it raises a clear error. The heavy ``torch`` / model imports are
deferred to construction time, so importing the package (or using the classical detector) never pays
for them.
"""

from __future__ import annotations

from typing import ClassVar

from maneuver_detect.detectors.learned import _LearnedDetector

__all__ = ["CHECKPOINT_ENV", "TransformerDetector"]

#: Environment variable naming a checkpoint-bundle path the no-argument detector loads, so the
#: ``detect(history, model="transformer-base")`` dispatch path can find a trained model without the
#: caller threading one through. A published-checkpoint default (Hub) arrives in a later release.
CHECKPOINT_ENV = "MANEUVER_DETECT_TRANSFORMER_CHECKPOINT"


class TransformerDetector(_LearnedDetector):
    """Learned transformer detector — per-gap localisation by the model, Δv/type by the physics.

    Construct with a trained checkpoint (a :class:`~maneuver_detect.models.checkpoint.ModelBundle`
    or a path to one); the no-argument construction the registry uses falls back to the
    :data:`CHECKPOINT_ENV` path, and raises from :meth:`detect` if neither is available.
    ``threshold`` overrides the bundle's default per-gap maneuver-probability threshold. All the
    inference machinery is inherited from the shared
    :class:`~maneuver_detect.detectors.learned._LearnedDetector`.
    """

    name = "transformer-base"
    checkpoint_env: ClassVar[str] = CHECKPOINT_ENV
