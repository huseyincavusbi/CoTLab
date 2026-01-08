# Prompt Strategies

## General Prompts

Located in `conf/prompt/`:

- `chain_of_thought` - Step-by-step reasoning
- `direct_answer` - Answer only
- `sycophantic` - Suggest wrong answer
- `adversarial` - Challenge the model
- `uncertainty` - Confidence levels
- `expert_persona` - Specialist personas
- `few_shot` - Include examples

## Medical Specialty Prompts

| Prompt | Task |
|--------|------|
| `radiology` | Pathological fracture detection |
| `cardiology` | Congenital heart defect detection |
| `neurology` | Neurological abnormality detection |
| `oncology` | Malignancy detection |

## Usage

```bash
python -m cotlab.main prompt=radiology dataset=radiology
python -m cotlab.main prompt=cardiology dataset=cardiology
```
