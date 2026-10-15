"""Download public course/Bitext data, then build disjoint reproducible splits."""
from __future__ import annotations
import hashlib
import json
import random
import re
import urllib.request
import pandas as pd
from src.common import DATA, CONFIG, canonical, clean_text, group_id, priority_for, write_json

COURSE = "https://raw.githubusercontent.com/SDAIAAcademy/applied-nlp/main/labs/data/"
BITEXT_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
BITEXT_URL = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/" + BITEXT_FILE
KB_NAMES = ["account", "cancellations", "contact", "feedback", "invoices", "newsletter", "orders", "payments", "refunds", "shipping"]

def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())

def main():
    # A complete snapshot ships in the ZIP. Rebuild needs the internet only for missing source files.
    DATA.mkdir(exist_ok=True)
    for name in KB_NAMES:
        download(COURSE + f"kb/{name}.md", DATA / "kb" / f"{name}.md")
    for name in ["arabic_tickets.csv", "rag_eval.csv"]:
        download(COURSE + name, DATA / name)
    raw_file = DATA / "raw" / BITEXT_FILE
    download(BITEXT_URL, raw_file)
    raw = pd.read_csv(raw_file).rename(columns={"instruction": "text"})
    raw = raw.dropna(subset=["text", "intent", "category", "response"]).copy()
    raw["text"] = raw.text.map(clean_text)
    raw["group_id"] = raw.text.map(group_id)
    # Keep one example per canonical template, so placeholder changes cannot leak across splits.
    before = len(raw)
    raw = raw.drop_duplicates("group_id").reset_index(drop=True)
    rng = random.Random(CONFIG["seed"])
    def fill(text):
        values = {}
        def replacement(m):
            key = m.group(1).strip().lower()
            if key not in values:
                values[key] = f"A-{rng.randint(10000,99999)}" if "order" in key else ("INV-2024-00731" if "invoice" in key else "DemoMart")
            return values[key]
        return re.sub(r"\{\{(.*?)\}\}", replacement, text)
    raw["text"] = raw.text.map(fill)
    # Historical response text is displayed as a past example. Current policy comes only from the KB.
    raw["response"] = raw.response.str.replace(r"\{\{.*?\}\}", "[historical placeholder]", regex=True)
    raw["ticket_id"] = [f"EN-{i:06}" for i in range(len(raw))]
    raw["lang"], raw["dialect"] = "en", "english"
    mapping = raw.groupby("intent").category.first().to_dict()
    raw["priority"] = [priority_for(i,t) for i,t in zip(raw.intent,raw.text)]
    parts = {"train": [], "val": [], "test_en": []}
    leftovers = []
    counts = [CONFIG[k] for k in ["train_per_intent", "val_per_intent", "test_per_intent"]]
    for _, group in raw.groupby("intent", sort=True):
        group = group.sample(frac=1, random_state=CONFIG["seed"])
        if len(group) < sum(counts):
            raise ValueError("Too few templates for configured split sizes")
        offset = 0
        for split,n in zip(parts,counts):
            parts[split].append(group.iloc[offset:offset+n]);offset += n
        leftovers.append(group.iloc[offset:])
    columns = ["ticket_id", "text", "intent", "category", "priority", "lang", "dialect", "group_id"]
    for split, groups in parts.items():
        pd.concat(groups)[columns].sample(frac=1,random_state=42).to_csv(DATA/f"{split}.csv",index=False)
    history = pd.concat(leftovers).sample(n=CONFIG["history_size"],random_state=42)
    history[columns+["response"]].to_csv(DATA/"history.csv",index=False)
    arabic = pd.read_csv(DATA/"arabic_tickets.csv")
    arabic["ticket_id"] = [f"AR-TEST-{i:04}" for i in range(len(arabic))]
    arabic["lang"], arabic["category"] = "ar", arabic.intent.map(mapping)
    arabic["group_id"] = arabic.text.map(group_id)
    arabic["priority"] = [priority_for(i,t) for i,t in zip(arabic.intent,arabic.text)]
    arabic[columns].to_csv(DATA/"test_ar.csv",index=False)
    # Separate authored Arabic TRAIN/VAL utterances. Never derive them from Arabic test tickets.
    extra = pd.read_csv(DATA/"arabic_authored.csv")
    for split in ["train", "val"]:
        df = extra[extra.split == split].copy()
        df["ticket_id"] = [f"AR-{split.upper()}-{i:04}" for i in range(len(df))]
        df["lang"],df["category"],df["dialect"] = "ar",df.intent.map(mapping),"authored_mixed"
        df["group_id"] = df.text.map(group_id)
        df["priority"] = [priority_for(i,t) for i,t in zip(df.intent,df.text)]
        df[columns].to_csv(DATA/f"{split}_ar.csv",index=False)
    frames = {s:pd.read_csv(DATA/f"{s}.csv") for s in ["train","val","test_en","test_ar","train_ar","val_ar","history"]}
    for i,(name,a) in enumerate(frames.items()):
        for other,b in list(frames.items())[i+1:]:
            if set(a.group_id) & set(b.group_id):
                raise ValueError(f"Canonical text overlap: {name}/{other}")
    write_json(DATA/"labels.json",{"intents":sorted(mapping),"intent_to_category":mapping})
    write_json(DATA/"manifest.json",{
        "seed":42,"source_url":BITEXT_URL,"source_sha256":hashlib.sha256(raw_file.read_bytes()).hexdigest(),
        "deduplicated_templates":before-len(raw),"sizes":{s:len(d) for s,d in frames.items()},
        "split_policy":"one canonical template per row, stratified disjoint splits, separate Arabic utterances",
        "snapshot_sha256":{f"{s}.csv":hashlib.sha256((DATA/f"{s}.csv").read_bytes()).hexdigest() for s in frames},
    })
    print(json.dumps({s:len(d) for s,d in frames.items()},indent=2))

if __name__ == "__main__":
    main()
