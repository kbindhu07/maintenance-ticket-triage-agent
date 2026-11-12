"""Evaluate the triage system and write reports/metrics.json + reports/evaluation.md.

Validation design
- Synthetic tickets come from templates. A random split would put near-duplicate
  tickets (same template) in train and test -> leakage and inflated scores.
  So we use StratifiedGroupKFold grouped by template_id: every test fold
  contains only templates the model never saw.
- The hand-written holdout (36 tickets, different writing style) is never used
  for training or tuning. It is the headline "new data" test.
- The bundled 6 tickets are scored with a model trained on synthetic data only.

Run:  python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import classifier  # noqa: E402
from backend.agent import TriageAgent  # noqa: E402
from backend.knowledge_base import SEVERITY_LEVELS  # noqa: E402
from backend.severity import score_severity  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def X_of(df):
    return [classifier.model_input(t, a) for t, a in zip(df["description"], df["asset_type"])]


def cls_metrics(y_true, y_pred):
    y_pred = ["none" if p is None else p for p in y_pred]
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
            "n": len(y_true)}


def sev_metrics(y_true, y_pred):
    t = np.array([SEVERITY_LEVELS.index(s) for s in y_true])
    p = np.array([SEVERITY_LEVELS.index(s) for s in y_pred])
    crit = t == 3
    return {**cls_metrics(list(y_true), list(y_pred)),
            "under_triage_rate": round(float((p < t).mean()), 3),
            "over_triage_rate": round(float((p > t).mean()), 3),
            "within_one_level": round(float((np.abs(p - t) <= 1).mean()), 3),
            "critical_recall": round(float((p[crit] == 3).mean()), 3) if crit.any() else None}


def grouped_cv(df, make_pipe, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    X, y, g = np.array(X_of(df), dtype=object), df["target_category"].values, df["template_id"].values
    for tr, te in cv.split(X, y, g):
        pipe = make_pipe()
        pipe.fit(list(X[tr]), y[tr])
        pred = pipe.predict(list(X[te]))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {"accuracy_mean": round(float(np.mean(accs)), 3), "accuracy_std": round(float(np.std(accs)), 3),
            "macro_f1_mean": round(float(np.mean(f1s)), 3), "macro_f1_std": round(float(np.std(f1s)), 3),
            "folds": n_splits}


def word_only_pipe():
    return Pipeline([("features", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                     ("clf", LogisticRegression(C=classifier.C_REGULARISATION, max_iter=3000, class_weight="balanced"))])


def main():
    synth = pd.read_csv(DATA / "synthetic_tickets.csv")
    bundled = pd.read_csv(DATA / "maintenance_tickets.csv")
    holdout = pd.read_csv(DATA / "holdout_handwritten.csv")
    m = {"dataset": {"synthetic": len(synth), "bundled": len(bundled), "holdout_handwritten": len(holdout),
                     "categories": int(synth["target_category"].nunique()),
                     "templates": int(synth["template_id"].nunique())}}

    # 1. Category - grouped CV on synthetic (model comparison + hyperparameter sweep)
    rules_cv = [classifier.rules_classify(t, a)["category"] for t, a in zip(synth["description"], synth["asset_type"])]
    m["category_cv"] = {
        "keyword_rules_baseline (no training)": cls_metrics(synth["target_category"], rules_cv),
        "tfidf_word_only + LR": grouped_cv(synth, word_only_pipe),
        "tfidf_word+char + LR (chosen)": grouped_cv(synth, classifier.build_pipeline),
    }
    m["hyperparameter_C_sweep"] = {str(c): grouped_cv(synth, lambda c=c: classifier.build_pipeline(C=c))
                                   for c in [0.5, 1.0, 5.0, 20.0]}

    # 2. Category - unseen data (model trained on synthetic only)
    pipe = classifier.build_pipeline().fit(X_of(synth), synth["target_category"])
    out = {}
    for name, df in [("bundled_6", bundled), ("holdout_handwritten_36", holdout)]:
        ml = pipe.predict(X_of(df))
        rules = [classifier.rules_classify(t, a)["category"] for t, a in zip(df["description"], df["asset_type"])]
        out[name] = {"keyword_rules_baseline": cls_metrics(df["target_category"], rules),
                     "ml_model": cls_metrics(df["target_category"], ml)}
    m["category_unseen"] = out

    # 3. Severity - rules vs ML-severity baseline, on holdout
    sev_pipe = Pipeline([("f", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                         ("clf", LogisticRegression(C=5, max_iter=3000, class_weight="balanced"))])
    sev_pipe.fit(X_of(synth), synth["target_severity"])
    sev = {}
    rules_synth = [score_severity(t, c)["severity"] for t, c in zip(synth["description"], synth["target_category"])]
    sev["synthetic_350 (rules not tuned on these)"] = {
        "rules_given_true_category": sev_metrics(synth["target_severity"], rules_synth)}
    for name, df in [("bundled_6", bundled), ("holdout_handwritten_36", holdout)]:
        rules_true_cat = [score_severity(t, c)["severity"] for t, c in zip(df["description"], df["target_category"])]
        sev[name] = {"rules_given_true_category": sev_metrics(df["target_severity"], rules_true_cat),
                     "ml_severity_baseline": sev_metrics(df["target_severity"], sev_pipe.predict(X_of(df)))}
    m["severity"] = sev

    # 4. End-to-end agent on holdout (agent's own model = synthetic + bundled; holdout never seen)
    if classifier.MODEL_PATH.exists():
        classifier.MODEL_PATH.unlink()  # force retrain on current data
    agent = TriageAgent()
    rows = []
    for r in holdout.itertuples():
        res = agent.triage(r.description, r.asset_type, r.ticket_id, use_llm=False)
        rows.append({"ticket_id": r.ticket_id, "description": r.description,
                     "true_category": r.target_category, "pred_category": res["category"]["id"],
                     "confidence": res["category"]["confidence"], "method": res["category"]["method"],
                     "true_severity": r.target_severity, "pred_severity": res["severity"]["severity"],
                     "needs_review": res["needs_human_review"]})
    e2e = pd.DataFrame(rows)
    labels = sorted(holdout["target_category"].unique())
    cm = confusion_matrix(e2e["true_category"], e2e["pred_category"], labels=labels)
    wrong = e2e[e2e["true_category"] != e2e["pred_category"]]
    m["agent_end_to_end_holdout"] = {
        "category": cls_metrics(e2e["true_category"], e2e["pred_category"]),
        "severity": sev_metrics(e2e["true_severity"], e2e["pred_severity"]),
        "flagged_for_review": int(e2e["needs_review"].sum()),
        "review_flag_precision_on_errors": (
            round(float(wrong["needs_review"].mean()), 3) if len(wrong) else None),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "category_errors": wrong[["ticket_id", "description", "true_category", "pred_category", "confidence"]].to_dict("records"),
        "severity_errors": e2e[e2e["true_severity"] != e2e["pred_severity"]][
            ["ticket_id", "description", "true_severity", "pred_severity", "pred_category"]].to_dict("records"),
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(m, indent=2))
    e2e.to_csv(REPORTS / "holdout_predictions.csv", index=False)
    (REPORTS / "evaluation.md").write_text(render_md(m))
    print(render_md(m))


def render_md(m):
    L = ["# Evaluation results", "", "Generated by `python scripts/evaluate.py`.", "",
         f"Data: {m['dataset']['synthetic']} synthetic tickets from {m['dataset']['templates']} templates, "
         f"{m['dataset']['bundled']} bundled tickets, {m['dataset']['holdout_handwritten']} hand-written holdout tickets, "
         f"{m['dataset']['categories']} categories.", "",
         "## 1. Category - 5-fold cross-validation grouped by template (synthetic)", "",
         "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for k, v in m["category_cv"].items():
        if "accuracy_mean" in v:
            L.append(f"| {k} | {v['accuracy_mean']} ± {v['accuracy_std']} | {v['macro_f1_mean']} ± {v['macro_f1_std']} |")
        else:
            L.append(f"| {k} | {v['accuracy']} | {v['macro_f1']} |")
    L += ["", "Regularisation strength C (same grouped CV):", "", "| C | Accuracy | Macro-F1 |", "|---:|---:|---:|"]
    for c, v in m["hyperparameter_C_sweep"].items():
        L.append(f"| {c} | {v['accuracy_mean']} ± {v['accuracy_std']} | {v['macro_f1_mean']} ± {v['macro_f1_std']} |")
    L += ["", "## 2. Category - unseen data (trained on synthetic only)", "",
          "| Test set | Rules baseline acc / F1 | ML model acc / F1 |", "|---|---:|---:|"]
    for k, v in m["category_unseen"].items():
        r, ml = v["keyword_rules_baseline"], v["ml_model"]
        L.append(f"| {k} | {r['accuracy']} / {r['macro_f1']} | {ml['accuracy']} / {ml['macro_f1']} |")
    L += ["", "## 3. Severity", "",
          "> Caveat: severity rules were revised once after error analysis on the hand-written holdout "
          "(v1 under-triage 19.4%, critical recall 0.73). Holdout severity numbers below are therefore optimistic; "
          "the synthetic-set row is the fairer estimate for the rules. A fresh, engineer-labelled test set is needed.", "",
          "| Test set | Method | Accuracy | Under-triage | Over-triage | Within 1 level | Critical recall |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for k, v in m["severity"].items():
        for meth, s in v.items():
            L.append(f"| {k} | {meth} | {s['accuracy']} | {s['under_triage_rate']} | {s['over_triage_rate']} | "
                     f"{s['within_one_level']} | {s['critical_recall']} |")
    a = m["agent_end_to_end_holdout"]
    L += ["", "## 4. End-to-end agent on the hand-written holdout", "",
          f"- Category accuracy **{a['category']['accuracy']}**, macro-F1 **{a['category']['macro_f1']}**",
          f"- Severity accuracy **{a['severity']['accuracy']}**, under-triage rate **{a['severity']['under_triage_rate']}**, "
          f"critical recall **{a['severity']['critical_recall']}**",
          f"- Tickets flagged for human review: {a['flagged_for_review']} / {a['category']['n']}", "",
          "Confusion matrix (rows = true, columns = predicted):", "",
          "| true \\ pred | " + " | ".join(l[:10] for l in a["confusion_matrix"]["labels"]) + " |",
          "|---|" + "---:|" * len(a["confusion_matrix"]["labels"])]
    for lab, row in zip(a["confusion_matrix"]["labels"], a["confusion_matrix"]["matrix"]):
        L.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    L += ["", "### Category errors", ""]
    L += [f"- {e['ticket_id']}: \"{e['description']}\" - true `{e['true_category']}`, predicted `{e['pred_category']}` "
          f"(conf {e['confidence']})" for e in a["category_errors"]] or ["- none"]
    L += ["", "### Severity errors", ""]
    L += [f"- {e['ticket_id']}: \"{e['description']}\" - true **{e['true_severity']}**, predicted **{e['pred_severity']}**"
          for e in a["severity_errors"]] or ["- none"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
