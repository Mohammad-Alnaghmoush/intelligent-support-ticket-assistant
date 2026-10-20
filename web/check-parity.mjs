/* Reproducible comparison against Python fixtures; not a new accuracy benchmark. */
import { readFile, writeFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { prepareText, answer, classify } from "./src/engine.js";
const fixture = JSON.parse(await readFile(process.argv[2], "utf8"));
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
const omit = new Set([
  "latency_ms",
  "classification_ms",
  "retrieval_ms",
  "generation_ms",
  "score",
  "confidence",
]);
function semantic(value) {
  if (Array.isArray(value)) return value.map(semantic);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .filter(([k]) => !omit.has(k))
        .map(([k, v]) => [k, semantic(v)]),
    );
  return value;
}
const report = {
  samples: fixture.samples.length,
  classifier_messages: fixture.classifier.vectors.length,
  decision_agreement: 0,
  python_vector_output_agreement: 0,
  onnx_vector_output_agreement: 0,
};
for (const item of fixture.samples) {
  assert.equal(prepareText(item.raw), item.text);
  for (const [key, count] of [
    ["torch_vector", "python_vector_output_agreement"],
    ["onnx_vector", "onnx_vector_output_agreement"],
  ]) {
    const actual = answer(model, history, kb, item[key], item.text);
    assert.ok(Math.abs(actual.confidence - item.expected.confidence) < 0.00002);
    assert.deepEqual(
      semantic(actual),
      semantic(item.expected),
      `${key}: ${item.raw}`,
    );
    for (let i = 0; i < 3; i++) {
      assert.ok(
        Math.abs(
          actual.kb_sections[i].score - item.expected.kb_sections[i].score,
        ) <= 0.00002,
      );
      assert.ok(
        Math.abs(
          actual.similar_tickets[i].score -
            item.expected.similar_tickets[i].score,
        ) <= 0.00002,
      );
    }
    report[count]++;
  }
}
fixture.classifier.vectors.forEach((vector, i) => {
  const out = classify(model, vector, fixture.classifier.lang[i]);
  assert.equal(out.candidate, fixture.classifier.predicted[i]);
  assert.equal(out.intent !== null, fixture.classifier.accepted[i]);
  report.decision_agreement++;
});
console.log(JSON.stringify(report, null, 2));
await writeFile(
  new URL("../eval/web_decision_report.json", import.meta.url),
  JSON.stringify(report, null, 2) + "\n",
);
