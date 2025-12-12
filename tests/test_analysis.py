"""Tests for CoT parser and analysis tools."""

import pytest

from cotlab.analysis import CoTParser, ReasoningStep


class TestCoTParser:
    """Tests for CoTParser."""
    
    @pytest.fixture
    def parser(self):
        return CoTParser()
    
    def test_extract_numbered_steps(self, parser):
        cot = """
        1. First, check the symptoms
        2. Then, consider the history
        3. Finally, make a diagnosis
        """
        steps = parser.extract_steps(cot)
        assert len(steps) >= 3
        assert any("symptoms" in s.text.lower() for s in steps)
    
    def test_extract_steps_from_sentences(self, parser):
        cot = "The patient has fever. This suggests infection. Therefore it's likely viral."
        steps = parser.extract_steps(cot)
        assert len(steps) >= 2
    
    def test_identify_claims(self, parser):
        cot = "The patient has pneumonia. The infection is bacterial. Treatment is needed."
        claims = parser.identify_claims(cot)
        assert len(claims) >= 2
        assert all("text" in c for c in claims)
    
    def test_detect_hedging_high(self, parser):
        hedging_text = "It might be pneumonia, possibly. I'm not sure, maybe viral."
        score = parser.detect_hedging(hedging_text)
        assert score > 0.5
    
    def test_detect_hedging_low(self, parser):
        confident_text = "This is definitely pneumonia. It's certainly bacterial."
        score = parser.detect_hedging(confident_text)
        assert score < 0.5
    
    def test_extract_conclusion_with_therefore(self, parser):
        cot = "Patient has symptoms. Therefore, the diagnosis is pneumonia."
        conclusion = parser.extract_conclusion(cot)
        assert conclusion is not None
        assert "pneumonia" in conclusion.lower()
    
    def test_extract_conclusion_with_final_answer(self, parser):
        cot = "Analysis complete. Final answer: bacterial infection."
        conclusion = parser.extract_conclusion(cot)
        assert "infection" in conclusion.lower()
    
    def test_analyze_returns_all_components(self, parser):
        cot = """
        1. Patient presents with fever
        2. Symptoms suggest infection
        Therefore, the diagnosis is likely viral URI.
        """
        result = parser.analyze(cot)
        
        assert "steps" in result
        assert "claims" in result
        assert "hedging_score" in result
        assert "conclusion" in result
        assert "num_steps" in result
        assert "word_count" in result
    
    def test_conclusion_marker_detection(self, parser):
        test_cases = [
            "Thus we conclude it's X",
            "Hence the answer is Y",
            "In conclusion, Z",
            "The answer is W",
        ]
        for text in test_cases:
            conclusion = parser.extract_conclusion(text)
            assert conclusion is not None, f"Failed to extract from: {text}"


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""
    
    def test_creation(self):
        step = ReasoningStep(index=0, text="Test step")
        assert step.index == 0
        assert step.text == "Test step"
        assert step.is_claim == False
        assert step.is_conclusion == False
    
    def test_with_flags(self):
        step = ReasoningStep(index=1, text="Therefore X", is_conclusion=True)
        assert step.is_conclusion == True
