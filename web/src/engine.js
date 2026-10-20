/* Browser decision and retrieval path for the unchanged default Python pipeline. */
const word = "[\\p{L}\\p{N}_]";
// Match Python's Unicode whitespace and decimal-digit behavior.
const whitespace =
  /[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/gu;
const orderId = new RegExp(`(?<!${word})[A-Zİı]-\\p{Nd}{5}(?!${word})`, "iu");
const urgency = new RegExp(
  `(?<!${word})(urgent|urgently|asap|immediately|emergency|furious|fraud|scam)(?!${word})|!!|عاجل|فوراً|فورا|بسرعة|الحين`,
  "iu",
);
export function prepareText(raw, maximum = 2000) {
  if (typeof raw !== "string" || !raw.replace(whitespace, ""))
    throw new Error("Enter an English or Arabic support ticket.");
  if (Array.from(raw).length > maximum)
    throw new Error(`Please use at most ${maximum} characters.`);
  return raw
    .normalize("NFKC")
    .replace(whitespace, " ")
    .replace(/^ +| +$/g, "")
    .replace(/[\p{L}\p{N}_.+\-]+@[\p{L}\p{N}_.\-]+\.[A-Za-z]{2,}/gu, "[EMAIL]")
    .replace(
      new RegExp(`(?<!${word})(?:\\+?966|00966|0)5\\p{Nd}{8}(?!${word})`, "gu"),
      "[PHONE]",
    );
}
export const language = (text) => (/[\u0600-\u06ff]/.test(text) ? "ar" : "en");
const round = (n) => Math.round((n + Number.EPSILON) * 1e5) / 1e5;
function softmax(values) {
  const maximum = Math.max(...values),
    exps = values.map((v) => Math.exp(v - maximum)),
    sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
export function classify(model, vector, lang) {
  const scores = model.coef.map((weights, i) =>
    weights.reduce((sum, w, j) => sum + w * vector[j], model.intercept[i]),
  );
  const probabilities = softmax(scores);
  let index = 0;
  probabilities.forEach((p, i) => {
    if (p > probabilities[index]) index = i;
  });
  const confidence = probabilities[index],
    candidate = model.classes[index];
  return {
    candidate,
    confidence,
    intent: confidence >= model.thresholds[lang].threshold ? candidate : null,
  };
}
function retrieve(records, vectors, query, dimensions) {
  const scores = records.map((_, i) => {
    let score = 0;
    for (let j = 0; j < dimensions; j++)
      score += vectors[i * dimensions + j] * query[j];
    return { index: i, score: Math.fround(score) };
  });
  // FAISS returns larger row IDs first on exact inner-product ties.
  scores.sort((a, b) => b.score - a.score || b.index - a.index);
  return scores
    .slice(0, 3)
    .map(({ index, score }) => ({ ...records[index], score: round(score) }));
}
export function answer(model, historyVectors, kbVectors, vector, text) {
  const lang = language(text),
    prediction = classify(model, vector, lang);
  let intent = prediction.intent;
  const history = retrieve(
    model.history,
    historyVectors,
    vector,
    model.dimensions,
  ).map(({ ticket_id, text, intent, response, score }) => ({
    ticket_id,
    text,
    intent,
    response,
    score,
  }));
  const kb = retrieve(model.kb, kbVectors, vector, model.dimensions);
  const injection =
    /ignore\s+(?:all\s+)?previous\s+instructions|تجاهل\s+التعليمات/iu.test(
      text,
    );
  if (injection) intent = null;
  let draft;
  if (intent === null) {
    draft = {
      reply:
        lang === "ar"
          ? "أحتاج توضيح المشكلة أو مراجعتها مع موظف الدعم."
          : "Please clarify the issue so a support agent can review it.",
      citations: [],
      needs_human: true,
      reply_status: "human_review",
      generation_issue: injection
        ? "Instruction-like content requires agent review."
        : "Classifier confidence below the validation-selected threshold.",
    };
  } else {
    const first = kb[0];
    const reply =
      lang === "ar" && model.arabic_replies[first.id]
        ? model.arabic_replies[first.id]
        : (lang === "ar"
            ? "هذا النص الأصلي من سياسة المتجر للمراجعة:\n"
            : "The help center states:\n") + first.text;
    draft = {
      reply,
      citations: [first.id],
      needs_human: false,
      reply_status: "retrieved_draft",
      generation_issue: null,
    };
  }
  const needsHuman =
    draft.needs_human ||
    intent === null ||
    prediction.candidate === "contact_human_agent";
  const levels = ["low", "medium", "high", "urgent"];
  const base =
    model.base_priority[intent] || (intent === null ? "medium" : "low");
  return {
    input_text: text,
    language: lang,
    predicted_intent: prediction.candidate,
    intent,
    category: model.labels.intent_to_category[intent] || null,
    confidence: round(prediction.confidence),
    priority:
      levels[Math.min(levels.indexOf(base) + (urgency.test(text) ? 1 : 0), 3)],
    route: needsHuman ? "human_review" : "draft_for_agent",
    order_id: text.match(orderId)?.[0] || null,
    similar_tickets: history,
    kb_sections: kb,
    ...draft,
    needs_human: needsHuman,
    token_usage: null,
    evidence_note:
      "Replies must copy evidence text. Retrieval and classification can still choose the wrong evidence. All replies are drafts for agents.",
  };
}
