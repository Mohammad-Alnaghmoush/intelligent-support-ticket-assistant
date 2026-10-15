"""Evaluate all saved systems on the SAME held-out test and report honest slices."""
from __future__ import annotations
import json
import platform
import time
from collections import Counter
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from src.common import ARTIFACTS, CONFIG, DATA, EVAL, write_json
from src.embeddings import Embedder
from src.retrieval import Retriever

def bootstrap(y,p,labels,n=400):
    rng = np.random.default_rng(42); y,p = np.asarray(y),np.asarray(p)
    scores = []
    for _ in range(n):
        idx = rng.integers(0,len(y),len(y))
        scores.append(f1_score(y[idx],p[idx],labels=labels,average="macro",zero_division=0))
    return [float(v) for v in np.percentile(scores,[2.5,97.5])]

def main():
    EVAL.mkdir(exist_ok=True)
    en,ar = [pd.read_csv(DATA/f"{name}.csv") for name in ["test_en","test_ar"]]
    test = pd.concat([en,ar],ignore_index=True)
    train = pd.concat([pd.read_csv(DATA/"train.csv"),pd.read_csv(DATA/"train_ar.csv")],ignore_index=True)
    labels = json.loads((DATA/"labels.json").read_text())["intents"]
    meta = json.loads((ARTIFACTS/"training.json").read_text())
    embedder = Embedder();X = embedder.encode(test.text)
    trainX = embedder.encode(train.text)
    max_similarity = (X@trainX.T).max(axis=1)
    audit = test[["ticket_id","lang","text","intent"]].copy()
    audit["max_train_cosine"],audit["flag_above_095"] = max_similarity,max_similarity>0.95
    audit.to_csv(EVAL/"leakage_audit.csv",index=False)
    tables,all_pred,failures = [],[],[]
    for name in ["tfidf_en","e5_en","e5_bilingual"]:
        clf = joblib.load(ARTIFACTS/f"{name}.joblib")
        p = clf.predict_proba(test.text if name=="tfidf_en" else X)
        pred = clf.classes_[p.argmax(axis=1)];confidence = p.max(axis=1)
        accepted = np.ones(len(test),dtype=bool) if name=="tfidf_en" else np.array([c>=meta[name]["language_thresholds"][lang]["threshold"] for c,lang in zip(confidence,test.lang)])
        saved = test[["ticket_id","text","intent","lang","dialect"]].rename(columns={"intent":"y_true"}).copy()
        saved["model"],saved["y_pred"],saved["confidence"],saved["accepted"] = name,pred,confidence,accepted
        saved.to_csv(EVAL/f"predictions_{name}.csv",index=False);all_pred.append(saved)
        failures.append(saved[saved.y_true!=saved.y_pred])
        # Warm, single-query classifier timing includes vectorization/encoder, excludes loading and LLM.
        def predict_one(text):
            return clf.predict_proba([text] if name=="tfidf_en" else embedder.encode([text],cache=False))
        predict_one(test.text.iloc[0]);timings=[]
        for text in pd.concat([en.head(15),ar.head(15)]).text:
            start=time.perf_counter();predict_one(text);timings.append((time.perf_counter()-start)*1000)
        for slice_name,mask in [("all",np.ones(len(test),dtype=bool)),("en",(test.lang=="en").to_numpy()),("ar",(test.lang=="ar").to_numpy()),("msa",(test.dialect=="msa").to_numpy()),("gulf",(test.dialect=="gulf").to_numpy())]:
            if not mask.sum():continue
            y,pr = test.intent.to_numpy()[mask],pred[mask]
            acc_mask = mask & accepted
            low,high = bootstrap(y,pr,labels)
            clean = mask & (max_similarity<=0.95)
            tables.append({"model":name,"slice":slice_name,"n":int(mask.sum()),"macro_f1":f1_score(y,pr,labels=labels,average="macro",zero_division=0),
                "f1_ci_low":low,"f1_ci_high":high,"accuracy":accuracy_score(y,pr),"coverage":float(accepted[mask].mean()),
                "accepted_accuracy":float(accuracy_score(test.intent.to_numpy()[acc_mask],pred[acc_mask])) if acc_mask.sum() else None,
                "classifier_p95_ms":float(np.percentile(timings,95)),"api_fee_per_1000_usd":0.0,
                "without_near_duplicates_n":int(clean.sum()),"without_near_duplicates_macro_f1":float(f1_score(test.intent.to_numpy()[clean],pred[clean],labels=labels,average="macro",zero_division=0)) if clean.sum() else None})
        if name=="e5_bilingual":
            write_json(EVAL/"classification_report.json",classification_report(test.intent,pred,labels=labels,output_dict=True,zero_division=0))
            cm = confusion_matrix(test.intent,pred,labels=labels,normalize="true")
            fig,ax=plt.subplots(figsize=(13,11));im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=1)
            ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=8);ax.set_yticks(range(len(labels)),labels,fontsize=8)
            ax.set_xlabel("Predicted intent");ax.set_ylabel("True intent");ax.set_title("Bilingual e5 + LR: normalized held-out confusion matrix")
            fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(EVAL/"confusion_matrix.png",dpi=140);plt.close(fig)
    metrics=pd.DataFrame(tables);metrics.to_csv(EVAL/"classification_metrics.csv",index=False)
    candidates=pd.concat(failures,ignore_index=True);candidates.to_csv(EVAL/"failure_candidates.csv",index=False)
    # Retrieval uses the course's authored questions, including AR and two unanswerable queries.
    retriever=Retriever(embedder)
    questions=pd.concat([pd.read_csv(DATA/"rag_eval.csv").assign(partition="development"),
        pd.read_csv(DATA/"rag_test_authored.csv").assign(partition="test")],ignore_index=True).fillna("");rows=[]
    Q=embedder.encode(questions.question)
    for method in ["dense","hybrid"]:
        for n,row in questions.iterrows():
            ranked=retriever.search_kb(row.question,Q[n],k=len(retriever.kb),method=method)
            ids=[c["id"] for c in ranked]
            rank=(ids.index(row.gold_section)+1) if row.gold_section and row.gold_section in ids else None
            rows.append({"partition":row.partition,"method":method,"question":row.question,"lang":row.lang,"gold_section":row.gold_section,
                "answerable":bool(row.gold_section),"rank":rank,"hit_at_1":int(rank==1),"hit_at_3":int(rank is not None and rank<=3),
                "reciprocal_rank":1/rank if rank else 0,"top3":json.dumps(ids[:3])})
    rag=pd.DataFrame(rows);rag.to_csv(EVAL/"retrieval_predictions.csv",index=False)
    summaries=[]
    for partition in ["development","test"]:
        for method in ["dense","hybrid"]:
            for lang in ["all","en","ar"]:
                sample=rag[(rag.partition==partition)&(rag.method==method)&rag.answerable]
                if lang!="all":sample=sample[sample.lang==lang]
                summaries.append({"partition":partition,"method":method,"slice":lang,"n_answerable":len(sample),"hit_at_1":sample.hit_at_1.mean(),"hit_at_3":sample.hit_at_3.mean(),"mrr":sample.reciprocal_rank.mean()})
    pd.DataFrame(summaries).to_csv(EVAL/"retrieval_metrics.csv",index=False)
    write_json(EVAL/"evaluation_metadata.json",{"seed":42,"python":platform.python_version(),"platform":platform.platform(),
        "bootstrap_resamples":400,"bootstrap_note":"Percentile CI, fixed 27-label universe. Small slices can have downward-biased bootstrap intervals when labels disappear in resamples.",
        "classification_latency":"30 warm individual queries, 15 EN + 15 AR; excludes model load, retrieval and LLM generation",
        "cost_note":"0 external API fees; hardware, electricity and hosting costs are NOT zero and are not estimated.",
        "near_duplicate_flags":int((max_similarity>0.95).sum()),"template_limit":"Canonical duplicates removed. Semantic similarity flags remain and are reported, not a guarantee of full independence.",
        "failure_counts":{name:int(len(group)) for name,group in candidates.groupby("model")}})
    print(metrics[metrics.slice.isin(["all","en","ar"])][["model","slice","n","macro_f1","coverage","accepted_accuracy"]].to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Inspect eval/failure_candidates.csv and document at least 10 actual categorized errors.")

if __name__=="__main__":
    main()
