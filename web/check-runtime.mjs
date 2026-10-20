/* Exercise the same tokenizer, full-precision ONNX and WASM runtime as the worker. */
import { readFile, writeFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { prepareText, answer, classify } from "./src/engine.js";
import { queryInputs } from "./src/representation.js";
// Bundle browser utilities to exclude the native Node ONNX binding entirely.
await build({
  stdin: {
    contents:
      'export {XLMRobertaTokenizer,Tensor,mean_pooling} from "@huggingface/transformers";',
    resolveDir: fileURLToPath(new URL(".", import.meta.url)),
  },
  outfile: fileURLToPath(
    new URL("./.checks/transformer-tools.mjs", import.meta.url),
  ),
  platform: "browser",
  format: "esm",
  bundle: true,
  logLevel: "error",
});
globalThis[Symbol.for("onnxruntime")] = ort;
const { XLMRobertaTokenizer, Tensor, mean_pooling } = await import(
  "./.checks/transformer-tools.mjs"
);
const fixture = JSON.parse(await readFile(process.argv[2], "utf8"));
const directory = process.argv[3];
const tokenizer = new XLMRobertaTokenizer(
  JSON.parse(await readFile(directory + "/tokenizer.json", "utf8")),
  JSON.parse(await readFile(directory + "/tokenizer_config.json", "utf8")),
);
// Verification reads local weights, tokenizer and WASM; it needs no network.
ort.env.wasm.wasmPaths = new URL(
  "./node_modules/onnxruntime-web/dist/",
  import.meta.url,
).href;
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(
  await readFile(directory + "/onnx/model.onnx"),
  { executionProviders: ["wasm"] },
);
const model = JSON.parse(
  await readFile(
    new URL("./public/data/assistant.json", import.meta.url),
    "utf8",
  ),
);
async function vectors(name) {
  const b = await readFile(
    new URL("./public/data/" + name + ".f32", import.meta.url),
  );
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
const history = await vectors("history"),
  kb = await vectors("kb");
tokenizer.model_max_length = model.max_length;
const report = {
  runtime:
    "onnxruntime-web WASM, one thread; Transformers.js tokenizer and mean pooling",
  samples: fixture.samples.length,
  semantic_output_agreement: 0,
  max_embedding_absolute_difference: 0,
  max_probability_absolute_difference: 0,
  classifier_messages: fixture.classifier.texts.length,
  classifier_decision_agreement: 0,
  token_ids_agreement: 0,
};
async function embed(text, expectedTokens) {
  const tokens = queryInputs(tokenizer, Tensor, text, model.max_length);
  if (expectedTokens) {
    assert.deepEqual(Array.from(tokens.input_ids.data, Number), expectedTokens);
    report.token_ids_agreement++;
  }
  assert.ok(tokens.input_ids.dims[1] <= model.max_length);
  const shape = tokens.input_ids.dims,
    feeds = {};
  for (const name of session.inputNames)
    feeds[name] = new ort.Tensor(
      "int64",
      tokens[name]?.data || new BigInt64Array(shape[0] * shape[1]),
      shape,
    );
  const result = await session.run(feeds);
  const hidden = result.last_hidden_state;
  return mean_pooling(
    new Tensor(hidden.type, hidden.data, hidden.dims),
    tokens.attention_mask,
  ).normalize(2, -1).data;
}
for (const item of fixture.samples) {
  const text = prepareText(item.raw),
    pooled = await embed(text, item.tokens);
  report.max_embedding_absolute_difference = Math.max(
    report.max_embedding_absolute_difference,
    ...pooled.map((v, j) => Math.abs(v - item.torch_vector[j])),
  );
  const out = answer(model, history, kb, pooled, text),
    expected = item.expected;
  for (const key of [
    "input_text",
    "language",
    "predicted_intent",
    "intent",
    "category",
    "priority",
    "route",
    "order_id",
    "reply",
    "citations",
    "needs_human",
    "reply_status",
    "generation_issue",
  ])
    assert.deepEqual(out[key], expected[key], `${item.raw}: ${key}`);
  assert.deepEqual(
    out.kb_sections.map((k) => k.id),
    expected.kb_sections.map((k) => k.id),
  );
  assert.deepEqual(
    out.similar_tickets.map((k) => k.ticket_id),
    expected.similar_tickets.map((k) => k.ticket_id),
  );
  report.max_probability_absolute_difference = Math.max(
    report.max_probability_absolute_difference,
    Math.abs(out.confidence - expected.confidence),
  );
  assert.ok(Math.abs(out.confidence - expected.confidence) < 0.00002);
  report.semantic_output_agreement++;
  console.log("OK:", item.raw);
}
for (let i = 0; i < fixture.classifier.texts.length; i++) {
  const vector = await embed(fixture.classifier.texts[i]);
  const decision = classify(model, vector, fixture.classifier.lang[i]);
  assert.equal(decision.candidate, fixture.classifier.predicted[i]);
  assert.equal(decision.intent !== null, fixture.classifier.accepted[i]);
  report.classifier_decision_agreement++;
}
await session.release();
console.log(JSON.stringify(report, null, 2));
await writeFile(
  new URL("../eval/web_runtime_report.json", import.meta.url),
  JSON.stringify(report, null, 2) + "\n",
);
