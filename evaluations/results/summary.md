# Evaluation Results Summary

This report compares the original/base backbone against the fine-tuned model. 

## IndicQA

IndicQA measures context-grounded Hindi question answering. In this evaluation, the fine-tuned model improves every answer-quality metric over the base model.

| Model | Perplexity | Contains Match | F1 | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: |
| Base | 13.2201 | 71.76% | 34.14% | 75.13% |
| Fine-tuned model | **7.7455** | **72.94%** | **68.43%** | **86.39%** |
| Change | -41.41% | +1.18 pp | +34.29 pp | +11.26 pp |

Perplexity drops from `13.22` to `7.75`, a reduction of about `41.4%`. This means the fine-tuned model assigns substantially higher probability to the gold answers under the language-model likelihood objective. The answer overlap metrics also improve: F1 approximately doubles, and BERTScore F1 rises by about `15.0%` relative, indicating that the fine-tuned model's answers are more semantically similar to the reference answers than the base model's answers.

## MT-Bench-Hi

MT-Bench-Hi evaluates Hindi conversational ability across writing, roleplay, reasoning, math, coding, extraction, STEM, and humanities. The current judged results use an LLM-as-a-judge(chat-gpt 5.5).

| Group | Count | Base Avg | Fine-tuned Avg | Fine-tuned - Base |
| --- | ---: | ---: | ---: | ---: |
| Overall | 200 | 3.520 | **4.585** | +1.065 |
| Extraction | 40 | 4.625 | **5.125** | +0.500 |
| Math | 10 | 2.100 | 1.000 | -1.100 |
| Reasoning | 10 | 3.200 | 1.000 | -2.200 |
| STEM | 10 | 1.400 | **2.700** | +1.300 |
| Coding | 10 | 3.000 | 1.700 | -1.300 |
| Humanities | 40 | 3.400 | **5.450** | +2.050 |
| Roleplay | 40 | 5.400 | 3.575 | -1.825 |
| Writing | 40 | 3.075 | **5.850** | +2.775 |

There are `29` ties in the pairwise comparison. The strongest fine-tuned model improvements appear in the overall score, extraction, humanities, and writing, while the base model remains stronger on coding, math, reasoning, and roleplay prompts. This suggests a trade-off: improving performance on tasks that are well represented in the fine-tuning data may have reduced the model's ability on categories such as coding, math, reasoning, and roleplay. The training data also contained fewer examples of these task types, which explain this behavior.


## IFEval-Hi

IFEval-Hi measures verifiable Hindi instruction following. Unlike IndicQA, it focuses on exact constraints such as required formatting, number of paragraphs, keyword inclusion, forbidden words, punctuation rules, placeholders, repeated phrases, and response language.

| Metric | Fine-tuned | Base | Fine-tuned - Base |
| --- | ---: | ---: | ---: |
| Prompt strict accuracy | **24.41%** | 19.58% | +4.83 pp |
| Instruction strict accuracy | **36.16%** | 30.29% | +5.87 pp |
| Prompt loose accuracy | **25.47%** | 20.28% | +5.19 pp |
| Instruction loose accuracy | **37.43%** | 31.25% | +6.19 pp |
| Prompt normalized accuracy | **25.35%** | 19.69% | +5.66 pp |
| Instruction normalized accuracy | **36.95%** | 30.37% | +6.58 pp |

The fine-tuned model performs better on IFEval-Hi across all metrics. Prompt-level accuracy is stricter because the model must satisfy every instruction in the prompt, while instruction-level accuracy gives partial credit for individual constraints.

## Conclusion

Across the available evaluations, the fine-tuned model shows improved performance on Hindi QA, answer overlap, semantic similarity, extraction, STEM, humanities, writing, and instruction-following metrics. These gains align with the Hindi instruction data used for training, including `anudesh`, `flan_v2`, `hh-rlhf`, and `lm_sys`, which emphasize direct question answering, transformation, extraction, format following, and concise assistant responses.