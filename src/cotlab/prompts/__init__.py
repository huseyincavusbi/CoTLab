"""Prompt strategies module."""

from .cardiology import CardiologyPromptStrategy
from .length_matched_strategies import (
    ChainOfThoughtMatchedStrategy,
    ContrarianMatchedStrategy,
    DirectAnswerMatchedStrategy,
)
from .neurology import NeurologyPromptStrategy
from .oncology import OncologyPromptStrategy
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
    "NeurologyPromptStrategy",
    "OncologyPromptStrategy",
    "RadiologyPromptStrategy",
    "create_prompt_strategy",
    "ContrarianMatchedStrategy",
    "ChainOfThoughtMatchedStrategy",
    "DirectAnswerMatchedStrategy",
]
