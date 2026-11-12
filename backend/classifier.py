"""Category classifier (TF-IDF + Logistic Regression) and a keyword-rules baseline.

Why this model:
- Tickets are short free text with domain words ("gearbox", "pitch", "SCADA").
  TF-IDF captures those words; character n-grams make it robust to typos
  and abbreviations ("vib", "inv-4", "gbx").
- Logistic Regression is fast, works on a few hundred examples, gives
  calibrated-ish probabilities (used for the confidence gate in the agent) and
  its weights are directly explainable (we show the words that drove a decision).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .knowledge_base import CATEGORIES
from .preprocessing import clean_text, find_terms

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODEL_PATH = BASE / "models" / "ticket_classifier.joblib"

# --- Hyperparameters (explained in README) -----------------------------------
C_REGULARISATION = 5.0   # higher C = weaker regularisation = fits training words more closely
WORD_NGRAMS = (1, 2)     # single words + two-word phrases ("abnormal noise", "data missing")
CHAR_NGRAMS = (3, 5)     # sub-word pieces -> robust to typos / abbreviations
MIN_DF = 1               # keep rare words: the dataset is small


def model_input(text: str, asset_type: str = "") -> str:
    """The asset type is added as a token so the model can use it as context."""
    return f"asset_{(asset_type or 'unknown').lower()} {clean_text(text)}"


def build_pipeline(C: float = C_REGULARISATION) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=WORD_NGRAMS, min_df=MIN_DF, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAMS, min_df=MIN_DF, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def load_training_data() -> pd.DataFrame:
    """Bundled 6 tickets + synthetic training tickets. Hand-written holdout is NEVER used here."""
    frames = [pd.read_csv(DATA / "maintenance_tickets.csv").assign(source="bundled")]
    synth = DATA / "synthetic_tickets.csv"
    if synth.exists():
        frames.append(pd.read_csv(synth).assign(source="synthetic"))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["description", "target_category"])
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"] != ""].drop_duplicates(subset=["description"])
    df["asset_type"] = df["asset_type"].fillna("unknown").str.lower()
    return df.reset_index(drop=True)


def train(df: pd.DataFrame | None = None) -> Pipeline:
    df = load_training_data() if df is None else df
    X = [model_input(t, a) for t, a in zip(df["description"], df["asset_type"])]
    pipe = build_pipeline()
    pipe.fit(X, df["target_category"])
    return pipe


def train_severity(df: pd.DataFrame | None = None) -> Pipeline:
    """Secondary text model for severity. Used only as a conservative cross-check on the rules."""
    df = load_training_data() if df is None else df
    pipe = Pipeline([("features", TfidfVectorizer(ngram_range=WORD_NGRAMS, sublinear_tf=True)),
                     ("clf", LogisticRegression(C=C_REGULARISATION, max_iter=3000, class_weight="balanced"))])
    pipe.fit([model_input(t, a) for t, a in zip(df["description"], df["asset_type"])], df["target_severity"])
    return pipe


def load_or_train() -> dict:
    """Returns {'category': Pipeline, 'severity': Pipeline}. Retrains if the saved file is missing/incompatible."""
    if MODEL_PATH.exists():
        try:
            models = joblib.load(MODEL_PATH)
            if isinstance(models, dict) and {"category", "severity"} <= set(models):
                return models
        except Exception:
            pass  # incompatible pickle (e.g. different sklearn) -> retrain
    df = load_training_data()
    models = {"category": train(df), "severity": train_severity(df)}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    return models


def predict(pipe: Pipeline, text: str, asset_type: str = "", top_k: int = 3) -> dict:
    x = model_input(text, asset_type)
    proba = pipe.predict_proba([x])[0]
    classes = pipe.classes_
    order = np.argsort(proba)[::-1]
    top = [{"category": classes[i], "probability": round(float(proba[i]), 3)} for i in order[:top_k]]
    return {
        "category": classes[order[0]],
        "confidence": round(float(proba[order[0]]), 3),
        "margin": round(float(proba[order[0]] - proba[order[1]]), 3),
        "top_k": top,
        "evidence_terms": explain(pipe, x, classes[order[0]]),
    }


def explain(pipe: Pipeline, x: str, category: str, n: int = 6) -> list[str]:
    """Words in this ticket that pushed the prediction toward `category` the most."""
    word_vec = pipe.named_steps["features"].transformer_list[0][1]
    clf = pipe.named_steps["clf"]
    vec = word_vec.transform([x])
    n_word = len(word_vec.vocabulary_)
    class_idx = list(clf.classes_).index(category)
    coefs = clf.coef_[class_idx][:n_word]
    contrib = vec.multiply(coefs).tocsr()
    vocab = np.array(word_vec.get_feature_names_out())
    idx = contrib.indices[np.argsort(contrib.data)[::-1]]
    return [vocab[i] for i in idx if not vocab[i].startswith("asset_")][:n]


# --- Baseline -----------------------------------------------------------------
def rules_classify(text: str, asset_type: str = "") -> dict:
    """Keyword-matching baseline: count knowledge-base keywords per category."""
    t = clean_text(text)
    scores = {}
    for cat, info in CATEGORIES.items():
        hits = find_terms(info["keywords"], t)
        bonus = 0.5 if asset_type and asset_type.lower() in info["asset_types"] and hits else 0
        scores[cat] = (len(hits) + bonus, hits)
    best = max(scores, key=lambda c: scores[c][0])
    if scores[best][0] == 0:
        return {"category": None, "matched_keywords": [], "scores": {}}
    return {
        "category": best,
        "matched_keywords": scores[best][1],
        "scores": {c: s for c, (s, _) in scores.items() if s > 0},
    }
