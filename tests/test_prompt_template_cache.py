"""Regression test for prompt-template memoization with few_shot toggling.

activation_patching few_shot_contrast toggles ``strategy.few_shot`` per sample
to build clean (few_shot=True) vs corrupt (few_shot=False) prompts. The
memoized template cache MUST be keyed by the config attrs so the toggle
produces different prompts (clean != corrupt); a single cached template would
collapse all patching effects to denom=0.

Reference: src/cotlab/prompts/*.py (all 10 strategies with memoization:
_resolved_templates for the template-based ones, _few_shot_cache /
_format_instructions_cache for the MCQ-style ones, plus the lru_cached
generic few-shot block in strategies.py).
"""

import pytest

from cotlab.prompts.cardiology import CardiologyPromptStrategy
from cotlab.prompts.histopathology import HistopathologyPromptStrategy
from cotlab.prompts.mcq import MCQPromptStrategy
from cotlab.prompts.neurology import NeurologyPromptStrategy
from cotlab.prompts.oncology import OncologyPromptStrategy
from cotlab.prompts.plab import PLABPromptStrategy
from cotlab.prompts.pubmedqa import PubMedQAPromptStrategy
from cotlab.prompts.radiology import RadiologyPromptStrategy
from cotlab.prompts.tcga import TCGAPromptStrategy

STRATEGIES = [
    RadiologyPromptStrategy,
    CardiologyPromptStrategy,
    OncologyPromptStrategy,
    NeurologyPromptStrategy,
    HistopathologyPromptStrategy,
    TCGAPromptStrategy,
]

# MCQ-style: few_shot is a per-call build_prompt arg, examples cached by
# answer_first, format instructions cached by (output_format, answer_first).
MCQ_STYLE = [
    MCQPromptStrategy,
    PLABPromptStrategy,
    PubMedQAPromptStrategy,
]

_CACHES = ("_resolved_templates", "_few_shot_cache", "_format_instructions_cache")

# Only these four share the {report} PROMPT_TEMPLATE convention, so the
# literal per-call-path check applies to them; the rest are covered by the
# generic cache-clear identity + toggle tests above.
_TEMPLATE4 = [
    RadiologyPromptStrategy,
    CardiologyPromptStrategy,
    OncologyPromptStrategy,
    NeurologyPromptStrategy,
]


def _clear_caches(s):
    for name in _CACHES:
        cache = getattr(s, name, None)
        if isinstance(cache, dict):
            cache.clear()


@pytest.mark.parametrize("cls", STRATEGIES)
def test_few_shot_toggle_produces_different_prompts(cls):
    """Toggling few_shot must change the resolved template (no stale cache)."""
    s = cls(few_shot=True)
    clean = s.build_prompt({"text": "R"})
    s.few_shot = False
    corrupt = s.build_prompt({"text": "R"})
    s.few_shot = True
    clean_again = s.build_prompt({"text": "R"})
    assert clean != corrupt, f"{cls.__name__}: few_shot toggle had no effect"
    assert clean == clean_again, f"{cls.__name__}: cached template not idempotent"


@pytest.mark.parametrize("cls", MCQ_STYLE)
def test_mcq_style_few_shot_arg_toggle(cls):
    """few_shot is a per-call arg for MCQ-style strategies; toggling must differ."""
    s = cls()
    few = s.build_prompt({"text": "Q"}, few_shot=True)
    none = s.build_prompt({"text": "Q"}, few_shot=False)
    few_again = s.build_prompt({"text": "Q"}, few_shot=True)
    assert few != none, f"{cls.__name__}: few_shot arg had no effect"
    assert few == few_again, f"{cls.__name__}: cached examples not idempotent"


@pytest.mark.parametrize("cls", MCQ_STYLE)
def test_mcq_style_answer_first_toggle(cls):
    """Examples cache is keyed by answer_first: toggling must change the block."""
    s = cls()
    a = s._build_few_shot_examples()
    s.answer_first = not s.answer_first
    b = s._build_few_shot_examples()
    assert a != b, f"{cls.__name__}: answer_first toggle had no effect on examples"


@pytest.mark.parametrize("cls", MCQ_STYLE)
def test_cached_equals_rebuilt_after_cache_clear(cls):
    """Clearing the caches and rebuilding must be byte-identical (no stale/missing entries)."""
    s = cls(answer_first=True, output_format="json")
    first = s.build_prompt({"text": "Q"}, few_shot=True)
    _clear_caches(s)
    second = s.build_prompt({"text": "Q"}, few_shot=True)
    assert first == second, f"{cls.__name__}: cached output differs from rebuilt"


@pytest.mark.parametrize("cls", STRATEGIES)
def test_template_cache_clear_rebuild_identical(cls):
    """Clearing the template cache and rebuilding must be byte-identical."""
    s = cls(answer_first=True, contrarian=True, few_shot=True)
    first = s.build_prompt({"text": "R"})
    _clear_caches(s)
    second = s.build_prompt({"text": "R"})
    assert first == second, f"{cls.__name__}: cached output differs from rebuilt"


def test_mcq_format_instructions_keyed_by_output_format():
    """_format_instructions_cache key = (output_format, answer_first): toggles differ."""
    s = MCQPromptStrategy(output_format="json")
    j = s._get_format_instructions()
    s.output_format = "plain"
    p = s._get_format_instructions()
    s.answer_first = not s.answer_first
    p2 = s._get_format_instructions()
    assert j != p, "json vs plain instructions must differ"
    assert p != p2, "answer_first toggle must change plain instructions"


def test_strategies_generic_few_shot_block_idempotent():
    """Base _apply_prompt_flags / lru_cached block: idempotent, toggle-aware."""
    from cotlab.prompts.strategies import SimplePromptStrategy

    s = SimplePromptStrategy(few_shot=True)
    a = s.build_prompt({"text": "Q"})
    b = s.build_prompt({"text": "Q"})
    assert a == b, "repeated build with cached block must be identical"
    s.few_shot = False
    c = s.build_prompt({"text": "Q"})
    assert a != c, "few_shot toggle must change the prompt"
    assert "example" not in c.lower(), "no few-shot block when few_shot=False"


@pytest.mark.parametrize("cls", _TEMPLATE4)
def test_cached_template_matches_per_call_path(cls):
    """Cached resolution is byte-identical to the per-call transform."""
    from cotlab.prompts import cardiology as mod_c
    from cotlab.prompts import neurology as mod_n
    from cotlab.prompts import oncology as mod_o
    from cotlab.prompts import radiology as mod_r

    mod = {
        RadiologyPromptStrategy: mod_r,
        CardiologyPromptStrategy: mod_c,
        OncologyPromptStrategy: mod_o,
        NeurologyPromptStrategy: mod_n,
    }[cls]

    for af in [False, True]:
        for con in [False, True]:
            for fs in [False, True]:
                s = cls(contrarian=con, few_shot=fs, answer_first=af, output_format="json")
                if af:
                    t = mod.PROMPT_TEMPLATE_ANSWER_FIRST
                elif con:
                    t = mod.PROMPT_TEMPLATE_CONTRARIAN
                else:
                    t = mod.PROMPT_TEMPLATE
                if not fs:
                    t = s._remove_few_shot_examples(t)
                ref = t.format(report="R")
                got = s.build_prompt({"text": "R"})
                assert ref == got, f"{cls.__name__} fs={fs} af={af} con={con}"
