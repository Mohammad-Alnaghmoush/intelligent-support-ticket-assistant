"""Single end-to-end entry point: assist(text) -> a validated JSON-compatible dict."""
from __future__ import annotations
import argparse
import json
import re
import time
import joblib
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from src.common import ARTIFACTS, CONFIG, DATA, clean_text, language, mask_personal_data, priority_for
from src.embeddings import Embedder
from src.retrieval import Retriever
from src.generation import LocalGenerator, fallback

class TicketOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_text: str
    language: Literal["ar","en"]
    predicted_intent: str
    intent: str | None
    category: str | None
    confidence: float = Field(ge=0,le=1)
    priority: Literal["low","medium","high","urgent"]
    route: Literal["draft_for_agent","human_review"]
    order_id: str | None
    similar_tickets: list[dict]
    kb_sections: list[dict]
    reply: str
    citations: list[str]
    needs_human: bool
    reply_status: str
    generation_issue: str | None
    token_usage: dict | None = None
    latency_ms: float
    classification_ms: float
    retrieval_ms: float
    generation_ms: float
    evidence_note: str

class TicketAssistant:
    def __init__(self,use_llm=False,retrieval_method="dense"):
        model_file = ARTIFACTS/"e5_bilingual.joblib"
        if not model_file.exists():
            raise FileNotFoundError("Run python -m src.train before launching the assistant.")
        self.clf = joblib.load(model_file)
        metadata = json.loads((ARTIFACTS/"training.json").read_text())
        self.thresholds = metadata["e5_bilingual"]["language_thresholds"]
        self.labels = json.loads((DATA/"labels.json").read_text())
        self.embedder = Embedder()
        self.retriever = Retriever(self.embedder)
        self.generator = LocalGenerator() if use_llm else None
        self.retrieval_method = retrieval_method

    def __call__(self,text):
        if not isinstance(text,str) or not text.strip():
            raise ValueError("Enter a non-empty English or Arabic support ticket.")
        if len(text)>CONFIG["max_input_characters"]:
            raise ValueError(f"Please use at most {CONFIG['max_input_characters']} characters.")
        text = mask_personal_data(clean_text(text));lang = language(text)
        start = time.perf_counter()
        q = self.embedder.encode([text],cache=False)
        p = self.clf.predict_proba(q)[0];i = int(p.argmax())
        candidate,confidence = str(self.clf.classes_[i]),float(p[i])
        intent = candidate if confidence>=self.thresholds[lang]["threshold"] else None
        classify_end = time.perf_counter()
        similar = self.retriever.search_history(text,q)
        kb = self.retriever.search_kb(text,q,method=self.retrieval_method)
        retrieve_end = time.perf_counter()
        injection = bool(re.search(r"ignore\s+(?:all\s+)?previous\s+instructions|تجاهل\s+التعليمات",text,re.I))
        if injection:
            intent = None
            draft = fallback([],text,"Instruction-like content requires agent review.",False)
        elif intent is None:
            draft = fallback([],text,"Classifier confidence below the validation-selected threshold.",False)
        elif self.generator is None:
            draft = fallback(kb,text,"LLM disabled. Exact evidence is provided for agent review.")
            draft.update(reply_status="retrieved_draft",needs_human=False,generation_issue=None)
        else:
            try:
                draft = self.generator.draft(text,kb)
            except Exception as exc:
                draft = fallback(kb,text,f"{type(exc).__name__}: generation unavailable or output failed validation.")
        # Requesting a human is an explicit route, even if the intent classifier is certain.
        human = draft["needs_human"] or intent is None or candidate=="contact_human_agent"
        end = time.perf_counter()
        out = {"input_text":text,"language":lang,"predicted_intent":candidate,"intent":intent,
            "category":self.labels["intent_to_category"].get(intent),"confidence":round(confidence,5),
            "priority":priority_for(intent,text),"route":"human_review" if human else "draft_for_agent",
            "order_id":(m.group(0) if (m:=re.search(r"\b[A-Z]-\d{5}\b",text,re.I)) else None),
            "similar_tickets":similar,"kb_sections":kb,**draft,"needs_human":bool(human),
            "latency_ms":round((end-start)*1000,2),"classification_ms":round((classify_end-start)*1000,2),
            "retrieval_ms":round((retrieve_end-classify_end)*1000,2),"generation_ms":round((end-retrieve_end)*1000,2),
            "evidence_note":"Replies must copy evidence text. Retrieval and classification can still choose the wrong evidence. All replies are drafts for agents."}
        return TicketOutput.model_validate(out).model_dump()

_assistant = None
def assist(text: str) -> dict:
    global _assistant
    if _assistant is None:
        _assistant = TicketAssistant()
    return _assistant(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",required=True)
    parser.add_argument("--llm",action="store_true")
    args = parser.parse_args()
    print(json.dumps(TicketAssistant(use_llm=args.llm)(args.text),ensure_ascii=False,indent=2))
