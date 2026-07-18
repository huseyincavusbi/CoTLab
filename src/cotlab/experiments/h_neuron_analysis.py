"""H-Neuron Analysis Experiment.

DEPRECATED — Migrated to the standalone `hprobes` package.

H-Neuron identification and CETT-based analysis now lives at:

    GitHub:  https://github.com/huseyincavusbi/hprobes
    PyPI:    pip install hprobes

The full implementation has been removed from CoTLab to avoid divergence
from the canonical `hprobes` package. Use `hprobes` directly for all CETT,
H-Neuron, confabulation, and entropy-neuron overlap analysis.
"""

from __future__ import annotations

from typing import Any, Optional

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger

_REDIRECT_URL = "https://github.com/huseyincavusbi/hprobes"


@Registry.register_experiment("h_neuron_analysis")
class HNeuronAnalysisExperiment(BaseExperiment):
    """Deprecated — use `hprobes` package: pip install hprobes"""

    def __init__(self, name: str = "h_neuron_analysis", **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        msg = (
            f"\n{'=' * 60}\n"
            f"H-Neuron Analysis has moved to the `hprobes` package.\n"
            f"\n"
            f"  GitHub: {_REDIRECT_URL}\n"
            f"  PyPI:   pip install hprobes\n"
            f"\n"
            f"The full implementation was removed from CoTLab to avoid\n"
            f"divergence from the canonical version maintained in hprobes.\n"
            f"{'=' * 60}\n"
        )
        raise NotImplementedError(msg)
