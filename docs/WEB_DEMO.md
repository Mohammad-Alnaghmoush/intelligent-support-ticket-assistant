# Browser demo

Live application: https://mohammad-ticket-assistant.cool-dill-2655.chatgpt.site

The browser companion runs the default, LLM-disabled support-ticket pipeline. It supports English and Arabic messages, classification across 27 intents, category and priority rules, confidence-based review, three similar historical cases, three KB sections, cited reply drafts and JSON download. The interface labels remain English. Arabic input and approved Arabic replies remain supported.

## Preservation of model behavior

- Encoder: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, original `onnx/model.onnx`, FP32. No quantization or training.
- Representation: `query: ` prefix, 256-token truncation, attention-mask mean pooling, L2 normalization, 384 dimensions.
- Classifier: exported coefficients, intercepts and class order from `artifacts/e5_bilingual.joblib`.
- Thresholds: the original language-specific values from `artifacts/training.json`; visitors cannot alter them.
- Retrieval: exported FP32 historical and KB vectors from the original pipeline; normalized inner-product ranking replaces the FAISS implementation in JavaScript.
- Replies: the same original evidence text or source-linked approved Arabic reply. Low confidence and instruction-like messages follow the original review rules.

The Python core, trained artifacts, data and existing evaluation files are not changed. Optional Qwen and BM25/RRF remain Python experiments. Web timing measures the local browser and is not comparable to the existing reference CPU benchmark.

The port checks compare the same 513 held-out classification messages, plus ten full-pipeline examples covering Arabic, English, human requests, instruction-like inputs, masking, a long message and a known unrelated-query failure. Reports are saved in `eval/web_parity_report.json`, `eval/web_decision_report.json` and `eval/web_runtime_report.json`. These are parity checks, not a new independent accuracy study. Floating-point confidence/similarity values can differ in the last decimal places across runtimes; unchanged labels and review decisions on this finite set do not guarantee identical outputs for every possible input.

The recorded run has 513/513 matching raw labels and threshold decisions, including the WebAssembly runtime. Macro-F1 stays 0.9380906959. All ten example outputs match on intents, routes, priorities, replies, citations, retrieved KB IDs and historical-case IDs; their tokenizer IDs also match Python. Maximum probability difference is below 0.000001 before display rounding in the Python/ONNX comparison. One displayed example confidence differs by 0.00001 between runtimes. The existing nine behavioral tests also pass.

## Build from source

Install the repository's normal Python requirements first. Node.js 22 or newer is recommended. Run from the repository root:

```bash
python -m src.export_web
cd web
npm ci
npm run build
cd ..
python -m http.server 8000 --directory web/dist
```

Open `http://localhost:8000`. The export reads the saved classifier and original retrieval vectors; it does not retrain anything. Generated `web/public/data/`, `web/dist/` and `web/node_modules/` are intentionally excluded from Git. Generate the data before building. The resulting `web/dist/` is a self-contained static deployment directory, with model/tokenizer downloads pinned to the original Hugging Face revision. It can be deployed to a static host with HTTPS, including Vercel. No Python backend or secret is deployed.

The public application is hosted as a Site. Its source and public build were saved before deployment. GitHub contains the reproducible source; updating GitHub alone does not automatically rebuild the hosted application.

## Reproduce parity checks

After exporting the web data and running `npm ci`, run:

```bash
python -m pip install onnxruntime==1.31.0
python -m src.verify_web
```

This downloads the original FP32 ONNX export, compares it with the Python encoder/classifier, writes temporary fixtures under `artifacts/cache/`, checks JavaScript decisions and evidence against Python, and exercises the browser's ONNX WebAssembly runtime and Transformers.js tokenizer in Node. It does not launch a browser or measure browser responsiveness. It also checks the 256-token limit on a long input.

## Visitor requirements and limits

First use downloads about 470 MB of weights, plus the tokenizer and runtime. Model files may be reused from the browser cache. Loading time and speed depend on the device, memory, connection and browser. A recent desktop browser is recommended; low-memory mobile devices may fail to initialize the model. Reloading or clearing the browser cache may require another download.

Customer messages are analyzed locally in the worker. Network requests download public app/model files. The application does not submit tickets or execute customer-account actions. All replies remain support-agent drafts, and the project's documented classification and retrieval failures still apply.

Course: https://sdaiaacademy.github.io/applied-nlp/
