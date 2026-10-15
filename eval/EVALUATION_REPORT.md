# Evaluation report

All classifier systems use the SAME 513 held-out messages: 405 English Bitext messages and 108 Arabic course messages, spanning 27 intents. C is selected on EN/AR validation macro-F1 (equal language weight). Thresholds use in-domain validation plus 10 authored unrelated validation messages. Test labels do not select C or thresholds.

## Classification

| model | slice | n | macro_f1 | accuracy | coverage | accepted_accuracy | classifier_p95_ms |
|---|---|---|---|---|---|---|---|
| tfidf_en | all | 513 | 0.8481 | 0.7856 | 1.0000 | 0.7856 | 2.5728 |
| tfidf_en | en | 405 | 0.9851 | 0.9852 | 1.0000 | 0.9852 | 2.5728 |
| tfidf_en | ar | 108 | 0.0026 | 0.0370 | 1.0000 | 0.0370 | 2.5728 |
| e5_en | all | 513 | 0.8879 | 0.8869 | 0.8285 | 0.9859 | 16.2276 |
| e5_en | en | 405 | 0.9901 | 0.9901 | 0.9975 | 0.9926 | 16.2276 |
| e5_en | ar | 108 | 0.4748 | 0.5000 | 0.1944 | 0.8571 | 16.2276 |
| e5_bilingual | all | 513 | 0.9381 | 0.9376 | 0.8499 | 0.9977 | 16.3146 |
| e5_bilingual | en | 405 | 0.9951 | 0.9951 | 0.9975 | 0.9975 | 16.3146 |
| e5_bilingual | ar | 108 | 0.7249 | 0.7222 | 0.2963 | 1.0000 | 16.3146 |

Raw macro-F1 measures predictions before abstention. Coverage measures confidence acceptance. Accepted accuracy measures only accepted decisions, and is not the full-set accuracy. Classifier p95 uses 30 warm single-message calls (15 EN, 15 AR), includes representation, and excludes model loading, retrieval and reply drafting.

The bilingual head improves AR macro-F1 from 0.4748 to 0.7249, a 0.2501 absolute difference. It still rejects 70.4% of the Arabic test slice. Its 1.0000 accepted AR accuracy covers only 32 messages and is not a production guarantee. Results come from one seed, so we do not claim statistical significance from this run.

## Semantic leakage audit

| model | slice | without_near_duplicates_n | without_near_duplicates_macro_f1 |
|---|---|---|---|
| tfidf_en | all | 201 | 0.5160 |
| tfidf_en | en | 96 | 0.9468 |
| tfidf_en | ar | 105 | 0.0027 |
| e5_en | all | 201 | 0.6844 |
| e5_en | en | 96 | 0.9432 |
| e5_en | ar | 105 | 0.4697 |
| e5_bilingual | all | 201 | 0.8212 |
| e5_bilingual | en | 96 | 0.9432 |
| e5_bilingual | ar | 105 | 0.7145 |

312 of 513 test messages have maximum train cosine above 0.95. Most are templated English near-duplicates (309 EN, 3 AR). Exact canonical template groups are disjoint, but this semantic audit shows substantial remaining similarity. The filtered subset has 201 messages (96 EN, 105 AR). Its macro-F1 uses the fixed 27-label universe, including labels missing in a slice; the filtered combined value is therefore not directly comparable to the original combined value. Read the EN and AR slices too. These flags are not proof of actual contamination or perfect independence.

English scores are inflated by templated wording. Arabic training/validation data is authored and small. The real-world performance of this system remains unmeasured.

## Retrieval

| partition | method | slice | n_answerable | hit_at_1 | hit_at_3 | mrr |
|---|---|---|---|---|---|---|
| development | dense | all | 30 | 0.8667 | 0.9667 | 0.9103 |
| development | dense | en | 23 | 0.8696 | 1.0000 | 0.9203 |
| development | dense | ar | 7 | 0.8571 | 0.8571 | 0.8776 |
| development | hybrid | all | 30 | 0.7333 | 0.9333 | 0.8409 |
| development | hybrid | en | 23 | 0.6957 | 0.9565 | 0.8297 |
| development | hybrid | ar | 7 | 0.8571 | 0.8571 | 0.8776 |
| test | dense | all | 24 | 0.7917 | 0.8333 | 0.8497 |
| test | dense | en | 12 | 0.9167 | 0.9167 | 0.9375 |
| test | dense | ar | 12 | 0.6667 | 0.7500 | 0.7619 |
| test | hybrid | all | 24 | 0.7917 | 0.8750 | 0.8557 |
| test | hybrid | en | 12 | 0.8333 | 0.9167 | 0.8869 |
| test | hybrid | ar | 12 | 0.7500 | 0.8333 | 0.8244 |

The original 32 course questions serve as DEVELOPMENT: 30 answerable, 2 outside KB scope. Dense retrieval had higher development Hit@3 and MRR than BM25/RRF, so the FINAL system uses dense retrieval. The independently authored retrieval TEST has 24 answerable queries (12 EN, 12 AR) and 4 unrelated queries. Final dense test Hit@3 is 0.8333 and MRR is 0.8497. Hybrid happens to score higher test Hit@3 (0.8750), but we do not switch methods after inspecting test results.

Hit@k and MRR above use only answerable questions. Retrieval of an irrelevant section is not a correct answer for unanswerable queries. Full-path abstention is evaluated separately on saved samples and a scope smoke test. The small authored retrieval test is a teaching benchmark, not independent customer traffic.

## Reply drafting

The DEFAULT path copies retrieved English policy or uses an authored Arabic summary linked to the same source ID. The OPTIONAL Qwen path selects sentence numbers as validated structured output, then Python assembles exact evidence. It performs no free-form policy generation. Citation validity and exact evidence assembly ensure provenance, not relevance. Retrieval or classification can still select an inappropriate source.

Saved sample outputs contain both default and optional-model behavior. All responses are drafts for support agents. No customer actions or messages are sent. A literal instruction-injection phrase is routed to human review; this simple guard is not a comprehensive security evaluation.

## Cost and latency

No paid API is used: external API fee is 0 USD per 1000 messages. Hardware, electricity, downloads and hosting are not free and have not been priced. Model weights are external downloads, not in the ZIP. End-to-end timings and first-request latency are separate JSON files and depend on hardware. A seven-message warm p95 mixes routes and is only a smoke benchmark.

## Error analysis

The final classifier makes 32 raw-label mistakes. `error_analysis.csv` reviews 12 ACTUAL mistakes: 6 intent-boundary errors, 3 dialect-coverage errors, 1 typo case, 1 ambiguous wording, and 1 transliteration case. These causal tags are reviewer interpretations rather than causal experiments. Most reviewed mistakes are caught by abstention; one accepted invoice mistake remains, so human oversight still matters.

## Reproducibility

Seed 42. Model revisions and hyperparameters: `configs/project.json`. Source/split hash: `data/manifest.json`. Classification predictions, bootstrap percentile intervals (400 resamples), language/dialect slices and normalized confusion matrix ship with this report. Bootstrap CIs use a fixed 27-label universe and may shift downward on small slices when classes are absent in resamples. Treat them as descriptive uncertainty, not a significance test.

## Optional model smoke review

Three sampled tickets reached Qwen. All three produced valid structured evidence selections, and all three replies preserved source text. Only one selection answered the requested detail in the qualitative review: the Visa query selected Wallet timing, and the Arabic deletion query selected retained-record details without the deletion steps. This 1/3 observation is a small qualitative smoke review, not a population metric. It supports the default full-source draft instead of letting this tiny selector decide what policy details to omit. See `generation_review.json`.

## Out-of-domain failure observed

The default full-path smoke test rejected 5 of 6 unrelated queries. It failed to reject "What is the orbital period of Mars?": the closed-set classifier associated "period" with a supported intent and assigned confidence 0.9249. It produced an irrelevant support-policy draft. Confidence thresholding is not a complete novelty detector. The failure is recorded in `eval/scope_smoke.json`; a dedicated scope/answerability detector and a new independent test are future work.

Comparison note: this is a whole-system comparison, not an isolated data ablation. Validation selects C=5 for English-only e5 and C=50 for bilingual e5, so both language coverage and head regularization differ.
