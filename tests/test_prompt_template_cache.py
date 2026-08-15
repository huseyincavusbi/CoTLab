"""Regression test for prompt-template memoization with few_shot toggling.

activation_patching few_shot_contrast toggles ``strategy.few_shot`` per sample
to build clean (few_shot=True) vs corrupt (few_shot=False) prompts. The
memoized template cache MUST be keyed by the config attrs so the toggle
produces different prompts (clean != corrupt); a single cached template would
collapse all patching effects to denom=0.

Reference: src/cotlab/prompts/{radiology,cardiology,oncology,neurology}.py.
"""

import pytest

from cotlab.prompts.cardiology import CardiologyPromptStrategy
from cotlab.prompts.neurology import NeurologyPromptStrategy
from cotlab.prompts.oncology import OncologyPromptStrategy
from cotlab.prompts.radiology import RadiologyPromptStrategy

STRATEGIES = [
    RadiologyPromptStrategy,
    CardiologyPromptStrategy,
    OncologyPromptStrategy,
    NeurologyPromptStrategy,
]


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


@pytest.mark.parametrize("cls", STRATEGIES)
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
