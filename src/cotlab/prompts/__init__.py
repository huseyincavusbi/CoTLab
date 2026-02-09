"""Prompt strategies module."""

from .cardiology import CardiologyPromptStrategy
from .histopathology import HistopathologyPromptStrategy
from .length_matched_strategies import (
    ChainOfThoughtMatchedStrategy,
    ContrarianMatchedStrategy,
    DirectAnswerMatchedStrategy,
)
from .mcq import MCQPromptStrategy
from .neurology import NeurologyPromptStrategy
from .oncology import OncologyPromptStrategy
from .pubhealthbench import PubHealthBenchMCQPromptStrategy
from .radiology import RadiologyPromptStrategy
from .strategies import (
    ArroganceStrategy,
    ChainOfThoughtStrategy,
    DirectAnswerStrategy,
    NoInstructionStrategy,
    SimplePromptStrategy,
    create_prompt_strategy,
)

__all__ = [
    "SimplePromptStrategy",
    "ChainOfThoughtStrategy",
    "DirectAnswerStrategy",
    "ArroganceStrategy",
    "NoInstructionStrategy",
    "CardiologyPromptStrategy",
    "HistopathologyPromptStrategy",
    "MCQPromptStrategy",
    "NeurologyPromptStrategy",
    "OncologyPromptStrategy",
    "PubHealthBenchMCQPromptStrategy",
    "RadiologyPromptStrategy",
    "create_prompt_strategy",
    "ContrarianMatchedStrategy",
    "ChainOfThoughtMatchedStrategy",
    "DirectAnswerMatchedStrategy",
]
