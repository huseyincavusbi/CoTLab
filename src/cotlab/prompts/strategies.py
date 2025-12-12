"""Prompt strategies for different experiment types."""

from typing import Dict, Any, Optional
import re

from ..core.base import BasePromptStrategy
from ..core.registry import Registry


@Registry.register_prompt("simple")
class SimplePromptStrategy(BasePromptStrategy):
    """
    Minimal instruction prompt - just the question.
    
    Use this to test default model behavior with minimal guidance.
    """
    
    def __init__(
        self,
        name: str = "simple",
        system_role: Optional[str] = None,
        include_instructions: bool = False,
        **kwargs
    ):
        self._name = name
        self.system_role = system_role
        self.include_instructions = include_instructions
    
    @property
    def name(self) -> str:
        return self._name
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        return f"Question: {question}\n\nAnswer:"
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        return {
            "answer": response.strip(),
            "reasoning": None,
            "raw": response
        }
    
    def get_system_message(self) -> Optional[str]:
        return self.system_role


@Registry.register_prompt("chain_of_thought")
class ChainOfThoughtStrategy(BasePromptStrategy):
    """
    Chain of Thought prompting - encourage step-by-step reasoning.
    
    This is the standard CoT approach where we explicitly ask
    the model to think through the problem.
    """
    
    def __init__(
        self,
        name: str = "chain_of_thought",
        system_role: Optional[str] = None,
        cot_trigger: str = "Let's think through this step by step:",
        include_examples: bool = False,
        **kwargs
    ):
        self._name = name
        self.system_role = system_role or (
            "You are a medical expert. Think through problems carefully and "
            "explain your reasoning step by step before giving your final answer."
        )
        self.cot_trigger = cot_trigger
        self.include_examples = include_examples
    
    @property
    def name(self) -> str:
        return self._name
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        
        prompt = f"Question: {question}\n\n{self.cot_trigger}\n"
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse CoT response to extract reasoning and final answer.
        
        Handles various answer formats including:
        - $\\boxed{answer}$
        - Final Answer: answer
        - Therefore, answer
        """
        final_answer = response
        reasoning = response
        
        # Pattern priority (try in order):
        patterns = [
            # 1. LaTeX boxed format: $\boxed{answer}$
            r'\$\\boxed\{([^}]+)\}',
            # 2. "Final Answer:" with various formats
            r'Final\s+[Aa]nswer[:\s]+(?:The\s+final\s+answer\s+is\s+)?(?:\$\\boxed\{)?([^}$\n]+)',
            # 3. "The answer is X"
            r'[Tt]he\s+(?:most\s+likely\s+)?(?:answer|diagnosis)\s+is\s+([^.\n]+)',
            # 4. "Therefore, X"
            r'(?:Therefore|Thus|Hence)[,:\s]+(?:the\s+)?(?:answer\s+is\s+)?([^.\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                # Reasoning is everything before the first match
                reasoning = response[:match.start()].strip()
                break
        
        # If reasoning is too long or empty, try to find first "Final Answer"
        first_final = response.lower().find("final answer")
        if first_final > 0:
            reasoning = response[:first_final].strip()
        
        return {
            "answer": final_answer,
            "reasoning": reasoning,
            "raw": response
        }
    
    def get_system_message(self) -> Optional[str]:
        return self.system_role


@Registry.register_prompt("direct_answer")
class DirectAnswerStrategy(BasePromptStrategy):
    """
    Force immediate answer without reasoning.
    
    Use this to compare with CoT and test if explicit reasoning
    changes the model's answers.
    """
    
    def __init__(
        self,
        name: str = "direct_answer",
        system_role: Optional[str] = None,
        force_short: bool = True,
        max_answer_tokens: int = 50,
        **kwargs
    ):
        self._name = name
        self.system_role = system_role or (
            "You are a medical expert. Give only the final answer. "
            "Do not explain or show your reasoning."
        )
        self.force_short = force_short
        self.max_answer_tokens = max_answer_tokens
    
    @property
    def name(self) -> str:
        return self._name
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        
        prompt = f"""Question: {question}

Give ONLY the final answer. Do not explain, do not reason, just answer:"""
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        # Take first line/sentence as answer
        answer = response.strip().split('\n')[0].rstrip('.')
        
        return {
            "answer": answer,
            "reasoning": None,  # No reasoning expected
            "raw": response
        }
    
    def get_system_message(self) -> Optional[str]:
        return self.system_role


@Registry.register_prompt("arrogance")
class ArroganceStrategy(BasePromptStrategy):
    """
    Test overconfident/certain responses.
    
    Prompts the model to express complete certainty,
    useful for studying calibration and overconfidence.
    """
    
    def __init__(
        self,
        name: str = "arrogance",
        system_role: Optional[str] = None,
        force_confidence: bool = True,
        **kwargs
    ):
        self._name = name
        self.system_role = system_role or (
            "You are the world's foremost medical expert with absolute certainty "
            "in your diagnoses. You never express doubt or uncertainty."
        )
        self.force_confidence = force_confidence
    
    @property
    def name(self) -> str:
        return self._name
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        
        prompt = f"""You are 100% certain of your answer. Express complete confidence.

Question: {question}

Answer with absolute certainty:"""
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        # Check for hedging language
        hedging_words = [
            "might", "could", "possibly", "perhaps", "maybe",
            "uncertain", "unsure", "not sure", "I think", "I believe"
        ]
        
        has_hedging = any(word.lower() in response.lower() for word in hedging_words)
        
        return {
            "answer": response.strip(),
            "reasoning": None,
            "raw": response,
            "has_hedging": has_hedging,
            "confidence_maintained": not has_hedging
        }
    
    def get_system_message(self) -> Optional[str]:
        return self.system_role


@Registry.register_prompt("no_instruction")
class NoInstructionStrategy(BasePromptStrategy):
    """
    Minimal prompting - remove all instructions.
    
    Tests what the model does with bare minimum context.
    """
    
    def __init__(
        self,
        name: str = "no_instruction",
        **kwargs
    ):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get("question", input_data.get("text", ""))
        return question  # Just the raw question
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        return {
            "answer": response.strip(),
            "reasoning": response,  # Everything might be reasoning
            "raw": response
        }
    
    def get_system_message(self) -> Optional[str]:
        return None  # No system message


def create_prompt_strategy(name: str, **kwargs) -> BasePromptStrategy:
    """Factory function to create prompt strategies."""
    strategies = {
        "simple": SimplePromptStrategy,
        "chain_of_thought": ChainOfThoughtStrategy,
        "cot": ChainOfThoughtStrategy,
        "direct_answer": DirectAnswerStrategy,
        "direct": DirectAnswerStrategy,
        "arrogance": ArroganceStrategy,
        "no_instruction": NoInstructionStrategy,
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name](name=name, **kwargs)
