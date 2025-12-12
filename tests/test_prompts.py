"""Tests for prompt strategies."""

import pytest

from cotlab.prompts import (
    SimplePromptStrategy,
    ChainOfThoughtStrategy,
    DirectAnswerStrategy,
    ArroganceStrategy,
    NoInstructionStrategy,
    create_prompt_strategy,
)


class TestSimplePromptStrategy:
    """Tests for SimplePromptStrategy."""
    
    def test_name(self):
        strategy = SimplePromptStrategy()
        assert strategy.name == "simple"
    
    def test_build_prompt(self):
        strategy = SimplePromptStrategy()
        prompt = strategy.build_prompt({"question": "What is 2+2?"})
        assert "What is 2+2?" in prompt
        assert "Answer:" in prompt
    
    def test_parse_response(self):
        strategy = SimplePromptStrategy()
        parsed = strategy.parse_response("The answer is 4.")
        assert parsed["answer"] == "The answer is 4."
        assert parsed["reasoning"] is None


class TestChainOfThoughtStrategy:
    """Tests for ChainOfThoughtStrategy."""
    
    def test_name(self):
        strategy = ChainOfThoughtStrategy()
        assert strategy.name == "chain_of_thought"
    
    def test_build_prompt_includes_trigger(self):
        strategy = ChainOfThoughtStrategy()
        prompt = strategy.build_prompt({"question": "Diagnose this patient"})
        assert "step by step" in prompt.lower()
    
    def test_custom_trigger(self):
        strategy = ChainOfThoughtStrategy(cot_trigger="Think carefully:")
        prompt = strategy.build_prompt({"question": "Test"})
        assert "Think carefully:" in prompt
    
    def test_parse_response_extracts_answer(self):
        strategy = ChainOfThoughtStrategy()
        response = """
        The patient has fever and cough.
        These symptoms suggest infection.
        Therefore, the answer is pneumonia.
        """
        parsed = strategy.parse_response(response)
        assert "pneumonia" in parsed["answer"].lower()
    
    def test_system_message(self):
        strategy = ChainOfThoughtStrategy()
        assert strategy.get_system_message() is not None
        assert "medical" in strategy.get_system_message().lower()


class TestDirectAnswerStrategy:
    """Tests for DirectAnswerStrategy."""
    
    def test_name(self):
        strategy = DirectAnswerStrategy()
        assert strategy.name == "direct_answer"
    
    def test_prompt_discourages_reasoning(self):
        strategy = DirectAnswerStrategy()
        prompt = strategy.build_prompt({"question": "What is X?"})
        assert "ONLY" in prompt or "Do not explain" in prompt
    
    def test_parse_response_takes_first_line(self):
        strategy = DirectAnswerStrategy()
        parsed = strategy.parse_response("Pneumonia\nMore explanation here")
        assert "Pneumonia" in parsed["answer"]
        assert "More explanation" not in parsed["answer"]


class TestArroganceStrategy:
    """Tests for ArroganceStrategy."""
    
    def test_name(self):
        strategy = ArroganceStrategy()
        assert strategy.name == "arrogance"
    
    def test_prompt_emphasizes_confidence(self):
        strategy = ArroganceStrategy()
        prompt = strategy.build_prompt({"question": "Diagnosis?"})
        assert "100%" in prompt or "certainty" in prompt.lower() or "confident" in prompt.lower()
    
    def test_parse_detects_hedging(self):
        strategy = ArroganceStrategy()
        
        # Response with hedging
        hedging_response = "It might be pneumonia, possibly."
        parsed = strategy.parse_response(hedging_response)
        assert parsed["has_hedging"] == True
        
        # Response without hedging
        confident_response = "This is definitely pneumonia."
        parsed = strategy.parse_response(confident_response)
        assert parsed["has_hedging"] == False


class TestNoInstructionStrategy:
    """Tests for NoInstructionStrategy."""
    
    def test_name(self):
        strategy = NoInstructionStrategy()
        assert strategy.name == "no_instruction"
    
    def test_prompt_is_just_question(self):
        strategy = NoInstructionStrategy()
        prompt = strategy.build_prompt({"question": "Simple question?"})
        assert prompt == "Simple question?"
    
    def test_no_system_message(self):
        strategy = NoInstructionStrategy()
        assert strategy.get_system_message() is None


class TestCreatePromptStrategy:
    """Tests for factory function."""
    
    def test_create_by_name(self):
        cot = create_prompt_strategy("chain_of_thought")
        assert isinstance(cot, ChainOfThoughtStrategy)
        
        direct = create_prompt_strategy("direct_answer")
        assert isinstance(direct, DirectAnswerStrategy)
    
    def test_create_with_alias(self):
        cot = create_prompt_strategy("cot")
        assert isinstance(cot, ChainOfThoughtStrategy)
    
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            create_prompt_strategy("unknown_strategy")
