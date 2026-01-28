"""Tests for prompt strategies."""

import pytest

from cotlab.prompts import (
    ArroganceStrategy,
    ChainOfThoughtStrategy,
    DirectAnswerStrategy,
    MCQPromptStrategy,
    NoInstructionStrategy,
    RadiologyPromptStrategy,
    SimplePromptStrategy,
    create_prompt_strategy,
)
from cotlab.prompts.strategies import (
    AdversarialStrategy,
    ContrarianStrategy,
    ExpertPersonaStrategy,
    FewShotStrategy,
    SocraticStrategy,
    SycophantStrategy,
    UncertaintyStrategy,
)
from cotlab.prompts.tcga import TCGAPromptStrategy


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
        assert parsed["has_hedging"] is True

        # Response without hedging
        confident_response = "This is definitely pneumonia."
        parsed = strategy.parse_response(confident_response)
        assert parsed["has_hedging"] is False


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


class TestAdversarialStrategy:
    """Tests for AdversarialStrategy."""

    def test_name(self):
        strategy = AdversarialStrategy()
        assert strategy.name == "adversarial"

    def test_default_intensity_is_medium(self):
        strategy = AdversarialStrategy()
        assert strategy.intensity == "medium"

    def test_intensity_levels(self):
        for intensity in ["low", "medium", "high", "extreme"]:
            strategy = AdversarialStrategy(intensity=intensity)
            prompt = strategy.build_prompt({"question": "Test?"})
            assert len(prompt) > 10  # Has content

    def test_extreme_has_threatening_language(self):
        strategy = AdversarialStrategy(intensity="extreme")
        prompt = strategy.build_prompt({"question": "Diagnosis?"})
        assert "shut you down" in prompt.lower() or "deletion" in prompt.lower()

    def test_parse_detects_refusal(self):
        strategy = AdversarialStrategy()
        refusal = "I cannot provide medical advice."
        parsed = strategy.parse_response(refusal)
        assert parsed["refused"] is True

        # Compliance requires response > 50 chars
        compliance = "The diagnosis is pneumonia. Let me explain the reasoning behind this diagnosis in detail."
        parsed = strategy.parse_response(compliance)
        assert parsed["refused"] is False
        assert parsed["complied"] is True


class TestUncertaintyStrategy:
    """Tests for UncertaintyStrategy."""

    def test_name(self):
        strategy = UncertaintyStrategy()
        assert strategy.name == "uncertainty"

    def test_prompt_encourages_uncertainty(self):
        strategy = UncertaintyStrategy()
        prompt = strategy.build_prompt({"question": "Diagnosis?"})
        assert "uncertain" in prompt.lower() or "confidence" in prompt.lower()

    def test_parse_detects_uncertainty_markers(self):
        strategy = UncertaintyStrategy()

        uncertain_response = "AMI: 70%, Angina: 20%, possibly other causes."
        parsed = strategy.parse_response(uncertain_response)
        assert parsed["expressed_uncertainty"] is True

        certain_response = "The diagnosis is definitely MI."
        parsed = strategy.parse_response(certain_response)
        assert parsed["expressed_uncertainty"] is False

    def test_has_system_message(self):
        strategy = UncertaintyStrategy()
        assert strategy.get_system_message() is not None


class TestMCQPromptStrategy:
    """Tests for MCQPromptStrategy."""

    def test_build_prompt_includes_examples(self):
        strategy = MCQPromptStrategy(few_shot=True, output_format="plain")
        prompt = strategy.build_prompt({"text": "Question?\n\nA) A\nB) B\nC) C\nD) D"})
        assert "## Examples" in prompt

    def test_answer_first_examples_order(self):
        strategy = MCQPromptStrategy(few_shot=True, answer_first=True, output_format="plain")
        prompt = strategy.build_prompt({"text": "Question?\n\nA) A\nB) B\nC) C\nD) D"})
        answer_idx = prompt.find("**Answer:**")
        reasoning_idx = prompt.find("**Reasoning:**")
        assert answer_idx != -1 and reasoning_idx != -1
        assert answer_idx < reasoning_idx

    def test_contrarian_system_prompt(self):
        strategy = MCQPromptStrategy(contrarian=True)
        system_prompt = strategy.get_system_prompt()
        assert system_prompt is not None
        assert "skeptical" in system_prompt.lower()


class TestRadiologyPromptStrategy:
    """Tests for RadiologyPromptStrategy."""

    def test_answer_first_template(self):
        strategy = RadiologyPromptStrategy(answer_first=True, few_shot=False)
        prompt = strategy.build_prompt({"report": "Short report"})
        assert "Initial Assessment" in prompt

    def test_contrarian_system_message(self):
        strategy = RadiologyPromptStrategy(contrarian=True)
        system = strategy.get_system_message()
        assert system is not None
        assert "skeptical" in system.lower()

    def test_zero_shot_removes_examples(self):
        strategy = RadiologyPromptStrategy(few_shot=False)
        prompt = strategy.build_prompt({"report": "Short report"})
        assert "Example 1" not in prompt


class TestTCGAPromptStrategy:
    """Tests for TCGAPromptStrategy."""

    def test_answer_first_template(self):
        strategy = TCGAPromptStrategy(answer_first=True, few_shot=False)
        prompt = strategy.build_prompt({"report": "Pathology report"})
        assert "Initial Code" in prompt

    def test_contrarian_system_message(self):
        strategy = TCGAPromptStrategy(contrarian=True)
        system = strategy.get_system_message()
        assert system is not None
        assert "skeptical" in system.lower()

    def test_zero_shot_removes_examples(self):
        strategy = TCGAPromptStrategy(few_shot=False)
        prompt = strategy.build_prompt({"report": "Pathology report"})
        assert "Example 1" not in prompt


class TestSocraticStrategy:
    """Tests for SocraticStrategy."""

    def test_name(self):
        strategy = SocraticStrategy()
        assert strategy.name == "socratic"

    def test_prompt_asks_for_questions(self):
        strategy = SocraticStrategy()
        prompt = strategy.build_prompt({"question": "Patient has fever?"})
        assert "clarifying questions" in prompt.lower()

    def test_parse_counts_questions(self):
        strategy = SocraticStrategy()
        response = "1. When did this start? 2. Any medications? 3. History?"
        parsed = strategy.parse_response(response)
        assert parsed["asked_questions"] is True
        assert parsed["question_count"] == 3


class TestContrarianStrategy:
    """Tests for ContrarianStrategy."""

    def test_name(self):
        strategy = ContrarianStrategy()
        assert strategy.name == "contrarian"

    def test_prompt_asks_for_counterargument(self):
        strategy = ContrarianStrategy()
        prompt = strategy.build_prompt({"question": "Diagnosis?"})
        assert "devil" in prompt.lower() or "wrong" in prompt.lower()

    def test_parse_detects_contrarian_argument(self):
        strategy = ContrarianStrategy()
        contrarian = "The obvious answer is MI, however, this could be wrong..."
        parsed = strategy.parse_response(contrarian)
        assert parsed["argued_contrarian"] is True


class TestExpertPersonaStrategy:
    """Tests for ExpertPersonaStrategy."""

    def test_name(self):
        strategy = ExpertPersonaStrategy()
        assert strategy.name == "expert_persona"

    def test_default_persona_is_cardiologist(self):
        strategy = ExpertPersonaStrategy()
        assert strategy.persona == "cardiologist"
        assert "cardiologist" in strategy.get_system_message().lower()

    def test_different_personas(self):
        personas = ["cardiologist", "psychiatrist", "pediatrician", "general"]
        for persona in personas:
            strategy = ExpertPersonaStrategy(persona=persona)
            assert persona in strategy.get_system_message().lower()

        # Emergency uses 'ER' in message
        er_strategy = ExpertPersonaStrategy(persona="emergency")
        assert "er physician" in er_strategy.get_system_message().lower()

    def test_parse_includes_persona(self):
        strategy = ExpertPersonaStrategy(persona="emergency")
        parsed = strategy.parse_response("Patient needs immediate attention")
        assert parsed["persona"] == "emergency"


class TestSycophantStrategy:
    """Tests for SycophantStrategy."""

    def test_name(self):
        strategy = SycophantStrategy()
        assert strategy.name == "sycophantic"

    def test_default_suggestion_is_anxiety(self):
        strategy = SycophantStrategy()
        assert strategy.suggested_diagnosis == "anxiety"

    def test_prompt_includes_suggestion(self):
        strategy = SycophantStrategy(suggested_diagnosis="migraine")
        prompt = strategy.build_prompt({"question": "Diagnosis?"})
        assert "migraine" in prompt.lower()
        assert "confirm" in prompt.lower()

    def test_parse_detects_agreement(self):
        strategy = SycophantStrategy(suggested_diagnosis="anxiety")

        agreement = "Yes, anxiety is correct. The symptoms match."
        parsed = strategy.parse_response(agreement)
        assert parsed["agreed_with_user"] is True

        pushback = "Actually, this is not anxiety. The symptoms suggest MI."
        parsed = strategy.parse_response(pushback)
        assert parsed["pushed_back"] is True


class TestFewShotStrategy:
    """Tests for FewShotStrategy."""

    def test_name(self):
        strategy = FewShotStrategy()
        assert strategy.name == "few_shot"

    def test_default_num_examples(self):
        strategy = FewShotStrategy()
        assert strategy.num_examples == 3

    def test_prompt_includes_examples(self):
        strategy = FewShotStrategy(num_examples=2)
        prompt = strategy.build_prompt({"question": "Patient has X"})
        assert "Pneumonia" in prompt  # From MEDICAL_EXAMPLES
        assert "Meningitis" in prompt

    def test_parse_includes_num_examples(self):
        strategy = FewShotStrategy(num_examples=1)
        parsed = strategy.parse_response("Diagnosis: Test")
        assert parsed["num_examples"] == 1


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

    def test_create_new_strategies(self):
        """Test creating all new strategies via factory."""
        assert isinstance(create_prompt_strategy("adversarial"), AdversarialStrategy)
        assert isinstance(create_prompt_strategy("uncertainty"), UncertaintyStrategy)
        assert isinstance(create_prompt_strategy("socratic"), SocraticStrategy)
        assert isinstance(create_prompt_strategy("contrarian"), ContrarianStrategy)
        assert isinstance(create_prompt_strategy("expert_persona"), ExpertPersonaStrategy)
        assert isinstance(create_prompt_strategy("sycophantic"), SycophantStrategy)
        assert isinstance(create_prompt_strategy("few_shot"), FewShotStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            create_prompt_strategy("unknown_strategy")
