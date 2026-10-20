const $ = (id) => document.getElementById(id);
const examples = {
  "ar-delete": "أرغب في حذف حسابي نهائياً",
  "en-refund": "How long does a refund to my Visa card take?",
  "ar-review": "تم خصم المبلغ مرتين من بطاقتي!!",
};
let worker,
  ready = false,
  busy = false,
  last = null,
  analyzeAfterLoad = false;
const pretty = (value) =>
  value
    ? value.replaceAll("_", " ").replace(/\b[a-z]/g, (c) => c.toUpperCase())
    : "Human review";
function status(text, kind = "") {
  $("status").textContent = text;
  $("status-dot").className = "status-dot " + kind;
}
function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function setBusy(value) {
  busy = value;
  $("analyze").disabled = value;
  document.querySelectorAll(".example").forEach((b) => (b.disabled = value));
}
function reset() {
  last = null;
  $("results").hidden = true;
  $("empty").hidden = false;
  $("language-badge").hidden = true;
  error("");
}
function element(tag, text, cls) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function evidenceCard(title, code, text) {
  const box = element("article", "", "evidence-card");
  box.append(element("h4", title), element("code", code), element("p", text));
  return box;
}
function render(out) {
  last = out;
  $("empty").hidden = true;
  $("results").hidden = false;
  $("language-badge").hidden = false;
  $("language-badge").textContent =
    out.language === "ar" ? "Arabic" : "English";
  const human = out.route === "human_review";
  $("route-banner").className = "route-banner" + (human ? " review" : "");
  $("route-symbol").textContent = human ? "↗" : "✓";
  $("route-title").textContent = human
    ? "Support-agent review needed"
    : "Reply draft ready";
  $("route-note").textContent = human
    ? "This message needs clarification or a support agent’s review."
    : "Review the cited evidence before sending this reply to a customer.";
  $("intent").textContent = pretty(out.intent);
  $("intent-code").textContent =
    out.intent || "Predicted: " + out.predicted_intent;
  $("confidence").textContent = (out.confidence * 100).toFixed(2) + "%";
  $("confidence-fill").style.width = out.confidence * 100 + "%";
  $("priority").textContent = pretty(out.priority);
  $("category").textContent = out.category || "Not accepted";
  $("reply").textContent = out.reply;
  $("sources").replaceChildren();
  if (out.citations.length) {
    for (const id of out.citations)
      $("sources").append(element("span", id, "source-pill"));
  } else
    $("sources").append(
      element(
        "span",
        "No source selected for this review response.",
        "source-caption",
      ),
    );
  $("evidence").replaceChildren(
    ...out.kb_sections.map((k) => evidenceCard(k.section, k.id, k.text)),
  );
  $("kb-count").textContent = out.kb_sections.length;
  $("cases").replaceChildren(
    ...out.similar_tickets.map((k) =>
      evidenceCard(pretty(k.intent), k.ticket_id, k.text),
    ),
  );
  $("case-count").textContent = out.similar_tickets.length;
  $("raw").textContent = JSON.stringify(out, null, 2);
  $("copy").textContent = "Copy reply";
}
function makeWorker() {
  worker = new Worker(new URL("./assistant-worker.js", import.meta.url), {
    type: "module",
  });
  worker.onerror = () => {
    ready = false;
    setBusy(false);
    error(
      "The model could not start in this browser. Try a recent desktop browser with enough available memory.",
    );
    status("Model unavailable");
    $("action-label").textContent = "Retry loading";
    worker.terminate();
    worker = null;
  };
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      $("progress").hidden = false;
      $("progress").value = data.progress;
      status("Downloading model · " + data.progress + "%", "busy");
    }
    if (data.type === "ready") {
      ready = true;
      $("progress").hidden = true;
      $("download-note").textContent =
        "Ready. Messages are analyzed on your device. Model files can be reused from your browser cache.";
      $("action-label").textContent = "Analyze ticket";
      status("Ready to analyze", "ready");
      setBusy(false);
      if (analyzeAfterLoad) {
        analyzeAfterLoad = false;
        analyze();
      }
    }
    if (data.type === "result") {
      render(data.output);
      setBusy(false);
      status("Analysis complete", "ready");
      $("action-label").textContent = "Analyze ticket";
    }
    if (data.type === "error") {
      setBusy(false);
      analyzeAfterLoad = false;
      $("progress").hidden = true;
      error(data.message);
      status(
        ready ? "Ready to try again" : "Could not load the model",
        ready ? "ready" : "",
      );
      $("action-label").textContent = ready
        ? "Analyze ticket"
        : "Retry loading";
    }
  };
}
function analyze() {
  if (busy) return;
  error("");
  const value = $("ticket").value;
  if (ready) {
    if (!value.trim()) {
      error("Enter a customer message or choose an example.");
      $("ticket").focus();
      return;
    }
    setBusy(true);
    status("Analyzing the ticket…", "busy");
    $("action-label").textContent = "Analyzing…";
    worker.postMessage({ type: "analyze", text: value });
  } else {
    if (!worker) makeWorker();
    analyzeAfterLoad = !!value.trim();
    setBusy(true);
    status("Loading the assistant…", "busy");
    $("action-label").textContent = "Loading assistant…";
    worker.postMessage({ type: "load" });
  }
}
$("analyze").addEventListener("click", analyze);
$("ticket").addEventListener("input", () => {
  $("count").textContent = Array.from($("ticket").value).length + " / 2000";
});
$("ticket").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    analyze();
  }
});
document.querySelectorAll(".example").forEach((button) =>
  button.addEventListener("click", () => {
    $("ticket").value = examples[button.dataset.example];
    $("ticket").dispatchEvent(new Event("input"));
    reset();
    $("ticket").focus();
  }),
);
$("clear").addEventListener("click", () => {
  if (busy) return;
  $("ticket").value = "";
  $("ticket").dispatchEvent(new Event("input"));
  reset();
  status(ready ? "Ready to analyze" : "Not loaded yet", ready ? "ready" : "");
  $("ticket").focus();
});
$("copy").addEventListener("click", async () => {
  if (last) {
    try {
      await navigator.clipboard.writeText(last.reply);
      $("copy").textContent = "Copied";
    } catch {
      error(
        "Copy is unavailable in this browser. Select and copy the reply text.",
      );
    }
  }
});
$("download").addEventListener("click", () => {
  if (!last) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(last, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "ticket-analysis.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
