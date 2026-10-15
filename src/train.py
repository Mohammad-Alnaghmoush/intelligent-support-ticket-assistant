"""Train lightweight classifiers on frozen e5 vectors. Select only on validation."""
from __future__ import annotations
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from src.common import ARTIFACTS, CONFIG, DATA, write_json
from src.embeddings import Embedder

def choose_threshold(y, p, classes, ood_confidence):
    confidence, predicted = p.max(axis=1), classes[p.argmax(axis=1)]
    candidates = []
    # Reject every authored out-of-domain validation example; do not tune on test queries.
    minimum = float(np.max(ood_confidence)+0.01)
    for tau in np.linspace(0.15,0.95,81):
        if tau<minimum:
            continue
        accepted = confidence >= tau
        if accepted.sum() >= 10:
            accuracy = accuracy_score(np.asarray(y)[accepted],predicted[accepted])
            if accuracy >= CONFIG["target_accepted_accuracy"]:
                candidates.append((float(accepted.mean()),float(tau),float(accuracy)))
    if not candidates:
        return {"threshold":0.95,"coverage":float((confidence>=0.95).mean()),"accepted_accuracy":None,
            "note":"No threshold met the target with at least 10 accepted validation examples."}
    coverage,tau,accuracy = max(candidates,key=lambda x:(x[0],-x[1]))
    return {"threshold":tau,"coverage":coverage,"accepted_accuracy":accuracy,"ood_validation_max_confidence":float(np.max(ood_confidence))}

def main():
    ARTIFACTS.mkdir(exist_ok=True)
    train_en,train_ar = [pd.read_csv(DATA/f"{n}.csv") for n in ["train","train_ar"]]
    val = pd.concat([pd.read_csv(DATA/"val.csv"),pd.read_csv(DATA/"val_ar.csv")],ignore_index=True)
    embedder = Embedder()
    t = time.perf_counter()
    X_en = embedder.encode(train_en.text)
    X_ar = embedder.encode(train_ar.text)
    V = embedder.encode(val.text)
    ood = pd.read_csv(DATA/"scope_val.csv")
    O = embedder.encode(ood.text)
    encoding_s = time.perf_counter()-t
    records = {}
    for name,df,X in [
        ("e5_en",train_en,X_en),
        ("e5_bilingual",pd.concat([train_en,train_ar]),np.vstack([X_en,X_ar]))]:
        best, trials = None, []
        for c in CONFIG["C_candidates"]:
            start = time.perf_counter()
            clf = LogisticRegression(C=c,max_iter=2000,class_weight="balanced",random_state=42).fit(X,df.intent)
            pred = clf.predict(V)
            # Equal-weight language score avoids letting the larger EN slice hide AR errors.
            score = float(np.mean([f1_score(val.loc[val.lang==lang,"intent"],pred[val.lang==lang],average="macro",zero_division=0) for lang in ["en","ar"]]))
            trial = {"C":c,"mean_language_macro_f1":score,"fit_seconds":time.perf_counter()-start}
            trials.append(trial)
            if best is None or score > best[0]:
                best = (score,clf,c)
        _,clf,c = best
        p = clf.predict_proba(V)
        op = clf.predict_proba(O).max(axis=1)
        thresholds = {lang:choose_threshold(val.loc[val.lang==lang,"intent"],p[val.lang==lang],clf.classes_,op[ood.lang==lang]) for lang in ["en","ar"]}
        joblib.dump(clf,ARTIFACTS/f"{name}.joblib")
        records[name] = {"C":c,"validation_candidates":trials,"language_thresholds":thresholds,"training_examples":len(df)}
        print(name,records[name],flush=True)
    t = time.perf_counter()
    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True),LogisticRegression(C=10,max_iter=2000,random_state=42))
    baseline.fit(train_en.text,train_en.intent)
    joblib.dump(baseline,ARTIFACTS/"tfidf_en.joblib")
    records["tfidf_en"]={"fit_seconds":time.perf_counter()-t,"note":"English-only lexical baseline. Fixed C=10 from course lab."}
    records["embedding_precompute_seconds"]=encoding_s
    records["config"]=CONFIG
    write_json(ARTIFACTS/"training.json",records)
    print("Models saved. Next: python -m src.evaluate",flush=True)

if __name__ == "__main__":
    main()
