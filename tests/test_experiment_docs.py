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
