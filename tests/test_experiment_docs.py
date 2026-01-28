"""Tests for experiment documentation generation."""

from omegaconf import OmegaConf

from cotlab.experiment import ExperimentDocumenter


def test_activation_compare_doc_content(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {
                "name": "activation_compare",
                "num_samples": 10,
                "seed": 42,
                "variants": [
                    {"name": "radiology", "dataset": "base", "prompt": "base"},
                    {
                        "name": "medqa",
                        "dataset": {
                            "name": "medqa",
                            "_target_": "cotlab.datasets.loaders.MedQADataset",
                        },
                        "prompt": {
                            "name": "mcq",
                            "_target_": "cotlab.prompts.mcq.MCQPromptStrategy",
                        },
                    },
                ],
            },
            "prompt": {"name": "radiology", "output_format": "plain", "few_shot": True},
            "dataset": {"name": "radiology"},
        }
    )

    doc = ExperimentDocumenter(cfg, tmp_path).create_initial_doc()

    assert "Activation Compare" in doc
    assert "residual stream activations" in doc
    assert "How does PLAIN output format" not in doc
    assert "- radiology: dataset=radiology, prompt=radiology" in doc
    assert "- medqa: dataset=medqa, prompt=mcq" in doc
    assert "experiment=activation_compare" in doc


def test_logit_lens_doc_content(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "logit_lens", "num_samples": 1, "seed": 7},
            "prompt": {"name": "histopathology", "output_format": "plain", "few_shot": False},
            "dataset": {"name": "histopathology"},
        }
    )

    doc = ExperimentDocumenter(cfg, tmp_path).create_initial_doc()

    assert "experiment=logit_lens" in doc
    assert "prompt.output_format=plain" in doc
    assert "How does PLAIN output format affect parsing and accuracy?" in doc


def test_activation_patching_doc_content(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "activation_patching", "num_samples": 5, "seed": 1},
            "prompt": {"name": "radiology", "output_format": "json", "few_shot": True},
            "dataset": {"name": "radiology"},
        }
    )

    doc = ExperimentDocumenter(cfg, tmp_path).create_initial_doc()

    assert "experiment=activation_patching" in doc
    assert "prompt=radiology" in doc
    assert "dataset=radiology" in doc


def test_attention_analysis_doc_content(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "attention_analysis", "num_samples": 3},
            "prompt": {"name": "mcq", "output_format": "plain", "few_shot": False},
            "dataset": {"name": "medqa"},
        }
    )

    doc = ExperimentDocumenter(cfg, tmp_path).create_initial_doc()

    assert "experiment=attention_analysis" in doc
    assert "prompt.output_format=plain" in doc
    assert "Does the model perform well without few-shot examples" in doc
