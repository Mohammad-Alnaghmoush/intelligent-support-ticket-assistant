# Data and model sources

This is an educational capstone for a fictional DemoMart store. No real customer personal data is included in the authored Arabic examples.

| Resource | Source | Use |
|---|---|---|
| English messages and historical responses | [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | Training, validation, English test, historical index |
| Arabic test messages | [Course data](https://github.com/SDAIAAcademy/applied-nlp/blob/main/labs/data/arabic_tickets.csv) | Untouched classification test messages |
| English knowledge base | [Course KB](https://github.com/SDAIAAcademy/applied-nlp/tree/main/labs/data/kb) | Fictional policy evidence |
| Course RAG questions | [rag_eval.csv](https://github.com/SDAIAAcademy/applied-nlp/blob/main/labs/data/rag_eval.csv) | Development benchmark used to choose dense vs hybrid |
| Arabic train/validation utterances | `data/arabic_authored.csv` | Small authored teaching data, not actual customer records |
| Arabic source-linked reply summaries | `data/approved_arabic_replies.json` | Conservative summaries of the original English KB |
| Independent retrieval questions | `data/rag_test_authored.csv` | Authored retrieval holdout, not used to pick the final method |
| Out-of-domain validation text | `data/scope_val.csv` | Threshold calibration, separate from classification test |
| e5 | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Frozen multilingual representation |
| Qwen | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Optional structured evidence selection |

The course identifies the Bitext data license as **CDLA-Sharing-1.0**. The dataset and course materials retain their upstream rights and licenses; the ZIP does not replace those terms with a project-wide license. The original repository did not expose a root LICENSE file at the path checked. Preserve this attribution and review upstream terms before public redistribution. Model weights are downloaded from their original repositories and are not bundled in this ZIP.

Frozen model revisions are in `configs/project.json`. Local source hashes and split hashes are in `data/manifest.json` and `data/SNAPSHOT_SHA256.json`. Source URLs using `main` may change in the future; use the bundled snapshot to reproduce this run.
