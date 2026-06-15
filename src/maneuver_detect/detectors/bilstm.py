"""The BiLSTM learned detector — localise gaps with the model, invert the physics for Δv/type.

``BiLstmDetector`` is the inference side of the first learned baseline. The whole inference
pipeline — load a checkpoint bundle, run the bare network over the V5-encoded series on **CPU**, and
turn the per-token maneuver probabilities into the canonical schema via the same Gauss inversion the
classical detector uses — is architecture-agnostic and lives in
:class:`~maneuver_detect.detectors.learned._LearnedDetector`. This module only pins the BiLSTM's
registry name and its checkpoint-path environment variable; the architecture is selected from the
bundle's ``network`` tag at load time, so the shared base rebuilds the bidirectional LSTM here and
the transformer for its sibling detector.

The detector registers under ``"bilstm-base"``. It resolves a trained checkpoint in this order: an
explicit bundle (or path) passed to the constructor, then the :data:`CHECKPOINT_ENV` environment
variable, then — for the no-argument construction the registry uses
(``detect(history, model="bilstm-base")``) — the Hub-published checkpoint, fetched on first use. The
heavy ``torch`` / model imports are deferred to construction time, so importing the package (or
using the classical detector) never pays for them.
"""

from __future__ import annotations

from typing import ClassVar

from maneuver_detect.detectors.learned import _detected_gaps, _LearnedDetector

__all__ = ["CHECKPOINT_ENV", "BiLstmDetector", "_detected_gaps"]

#: Environment variable naming a local checkpoint-bundle path the no-argument detector loads, so the
#: ``detect(history, model="bilstm-base")`` dispatch path can use a trained model without the caller
#: threading one through. Unset, the no-argument detector falls back to the Hub checkpoint.
CHECKPOINT_ENV = "MANEUVER_DETECT_BILSTM_CHECKPOINT"


class BiLstmDetector(_LearnedDetector):
    """Learned BiLSTM detector — per-gap localisation by the model, Δv/type by the physics.

    Construct with a trained checkpoint (a :class:`~maneuver_detect.models.checkpoint.ModelBundle`
    or a path to one); the no-argument construction the registry uses falls back to the
    :data:`CHECKPOINT_ENV` path, and raises from :meth:`detect` if neither is available.
    ``threshold`` overrides the bundle's per-gap threshold with one gate for every class, and
    ``class_thresholds`` overrides its per-class gates. All inference machinery is inherited from
    the shared :class:`~maneuver_detect.detectors.learned._LearnedDetector`.
    """

    name = "bilstm-base"
    checkpoint_env: ClassVar[str] = CHECKPOINT_ENV
