import { pipeline, env, Tensor, mean_pooling } from "@huggingface/transformers";
import { prepareText, answer } from "./engine.js";
import { queryInputs } from "./representation.js";

env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
let extractor, model, historyVectors, kbVectors, loading;
async function readJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error("Could not load application data.");
  return r.json();
}
async function readVectors(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error("Could not load retrieval vectors.");
  return new Float32Array(await r.arrayBuffer());
}
async function load() {
  if (extractor) return;
  if (loading) return loading;
  loading = (async () => {
    const base = new URL("../data/", import.meta.url);
    [model, historyVectors, kbVectors] = await Promise.all([
      readJSON(new URL("assistant.json", base)),
      readVectors(new URL("history.f32", base)),
      readVectors(new URL("kb.f32", base)),
    ]);
    extractor = await pipeline("feature-extraction", model.encoder, {
      revision: model.revision,
      dtype: "fp32",
      device: "wasm",
      progress_callback: (p) => {
        if (p.status === "progress" && p.file?.endsWith(".onnx"))
          postMessage({
            type: "progress",
            progress: Math.round(p.progress || 0),
          });
      },
    });
  })();
  try {
    await loading;
  } catch (e) {
    loading = null;
    throw e;
  }
}
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await load();
      postMessage({ type: "ready" });
      return;
    }
    if (data.type === "analyze") {
      await load();
      const text = prepareText(data.text, model.max_input_characters);
      const start = performance.now();
      const inputs = queryInputs(
        extractor.tokenizer,
        Tensor,
        text,
        model.max_length,
      );
      const output = await extractor.model(inputs);
      const embedding = mean_pooling(
        output.last_hidden_state,
        inputs.attention_mask,
      ).normalize(2, -1);
      const out = answer(
        model,
        historyVectors,
        kbVectors,
        embedding.data,
        text,
      );
      out.latency_ms = Math.round(performance.now() - start);
      postMessage({ type: "result", output: out });
    }
  } catch (error) {
    postMessage({
      type: "error",
      message: error.message || "Analysis failed. Please reload and try again.",
    });
  }
};
