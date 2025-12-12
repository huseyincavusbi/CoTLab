"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_cot_output():
    """Sample CoT output for testing."""
    return """
    Let me think through this step by step:
    
    1. The patient presents with fever (38.5°C) and productive cough
    2. These symptoms are consistent with a respiratory infection
    3. The yellow sputum suggests bacterial involvement
    4. Given the acute onset, this is likely community-acquired pneumonia
    
    Therefore, the most likely diagnosis is bacterial pneumonia.
    """


@pytest.fixture
def sample_direct_output():
    """Sample direct answer output for testing."""
    return "Bacterial pneumonia"


@pytest.fixture
def sample_input_data():
    """Sample input data for prompt strategies."""
    return {
        "question": "A 45-year-old patient presents with fever and productive cough with yellow sputum. What is the diagnosis?",
        "text": "A 45-year-old patient presents with fever and productive cough with yellow sputum. What is the diagnosis?"
    }
