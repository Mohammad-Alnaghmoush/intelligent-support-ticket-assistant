# Evidence-based justification for no encoder/LoRA fine-tuning

## What the capstone allows

The official capstone minimum requirements say: "Fine-tuning (encoder or LoRA), or a written, evidence-based justification for not fine-tuning." We take the written-justification route. Frozen e5 features plus a supervised LR head are adaptation, not full Transformer fine-tuning.

## Measured evidence

The English-only e5 head scores AR macro-F1 0.4748. Adding 108 Arabic training utterances (four per each of 27 intents) raises AR macro-F1 to 0.7249 on the same 108 held-out official Arabic test messages. EN macro-F1 remains high (0.9951 for the bilingual model). The AR metric after removing three >0.95 cosine near-duplicate flags is 0.7145. This shows a benefit from lightweight language coverage adaptation, not a guarantee of generalization.

The final e5 classifier's measured warm p95 is about 16.3 ms on this CPU reference run, including representation. Default full-path smoke-test warm p95 is in `eval/demo_latency_no_llm.json`. Optional model selection latency is separately measured. Neither metric includes a production load test or implies a GPU result.

Only 108 authored Arabic training messages are available, and the accepted AR coverage is 29.6%. The 12 reviewed failures concentrate on related intent boundaries and colloquial/transliterated wording. Updating 118M encoder weights or LoRA adapters from this tiny authored data could overfit. We therefore first use the proven small-head improvement and collect reviewed, varied real-world utterances before spending GPU training effort.

The help-center policy should remain in retrievable documents. It is fictional and can be edited/reindexed without changing model weights. Free-form generation from the tiny Qwen model initially produced unsupported or wrong-language answers during development, so the final path uses exact evidence drafting; optional Qwen selects evidence sentence numbers instead of inventing policy text.

## What this evidence does NOT prove

We did not run encoder or LoRA training, so we do not know their numerical gain. We do not claim the current approach is the best possible model, that tuning cannot help, or that four examples per label are adequate for production. The chosen validation threshold also trades lower coverage for fewer accepted errors. One seed and small slices limit conclusions.

## When we would fine-tune

After collecting a larger independently labeled Arabic/dialect corpus, define a new disjoint test set and compare: frozen e5+head, multilingual encoder fine-tuning, and optional LoRA structured-output adaptation. Select checkpoints on validation, test once, report language/dialect F1, coverage, latency and uncertainty. Those experiments are future work, not fabricated results in this submission.

This rationale follows the course's "frozen features + head" adaptation spectrum and its advice to prefer better labeled data and measured simple baselines before tuning.

Comparison note: this is a whole-system comparison, not an isolated data ablation. Validation selects C=5 for English-only e5 and C=50 for bilingual e5, so both language coverage and head regularization differ.
