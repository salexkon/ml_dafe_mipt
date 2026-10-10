"""HW1, задача 1 (реакция пользователя). Всё, что нужно ноутбукам: чтение, обработка текста,
модели, метрики, инференс. Инструменты — с семинаров 1–5: TF-IDF, стемминг nltk,
Naive Bayes, логистическая регрессия, SVM, SVD, GridSearchCV, Pipeline + joblib."""
import json, os

import joblib
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC

REACTIONS = ["negative", "neutral", "positive"]
MAX_CHARS = 2000                       # длинные вставки (код, эссе) режем: иначе они забивают словарь
ASST_HEAD_WORDS = 40 
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
_STEMMER = SnowballStemmer("english")


# ---------- чтение ----------
def read_dialogs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- обработка текста (семинар 4, раздел 10) ----------
_STEM_CACHE = {}

def _stem(word):
    if word not in _STEM_CACHE:
        _STEM_CACHE[word] = _STEMMER.stem(word)
    return _STEM_CACHE[word]


def clean_text(text):
    text = (text or "").replace("\\n", " ")
    return " ".join(text.split())[:MAX_CHARS]


def normalize(text):
    """Токены wordpunct -> нижний регистр -> основы слов. Стоп-слова НЕ убираем: в списке
    есть «not», «no», а для реакции отрицание важно (семинар 4). Знаки «!» и «?» оставляем:
    они несут эмоцию, а стандартный token_pattern sklearn их бы выбросил."""
    out = []
    for tok in wordpunct_tokenize(clean_text(text).lower()):
        if tok.isalpha():
            out.append(_stem(tok))
        elif tok in ("!", "?"):
            out.append(tok)
    return " ".join(out)


def to_frame(dialogs):
    """Диалоги -> таблица признаков. Один и тот же код идёт и в обучение, и в инференс."""
    rows = []
    for d in dialogs:
        turns = d["turns"]
        us = [t["text"] or "" for t in turns if t["role"] == "user"]
        asst = [t["text"] or "" for t in turns if t["role"] == "assistant"]
        all_messages = [t["text"] or "" for t in turns]
        follow = us[1:] if len(us) > 1 else us          # реакция живёт в ответах на ответы модели
        gkey = "".join(ch for ch in us[0].lower() if ch.isalnum())[:80] if us else d["dialog_id"]
        rows.append(dict(
            dialog_id=d["dialog_id"],
            text = normalize(" ".join(all_messages)),            # единый текст для базовых моделей
            all_user = normalize(" ".join(follow)),
            last_user = normalize(us[-1]) if us else "",
            first_user = normalize(us[0]) if us else "",
            all_asst = normalize(" ".join(asst)),
            last_asst = normalize(asst[-1]) if asst else "",
            last_asst_head = normalize(" ".join(asst[-1].split()[:ASST_HEAD_WORDS])) if asst else "",
            last_raw = clean_text(us[-1]) if us else "",   # сырой текст: для разбора ошибок
            n_user = len(us),
            len_last = len(clean_text(us[-1])) if us else 0,
            len_asst = float(np.mean([len(a) for a in asst])) if asst else 0.0,
            group = gkey,                                  # шаблонные/почти-дубли первой реплики
            reaction = (d.get("labels") or {}).get("reaction"),
        ))
    return pd.DataFrame(rows)


META_COLS = ["n_user", "len_last", "len_asst"]


def _log_meta(df):
    return np.log1p(df[META_COLS].to_numpy(dtype=float))


# ---------- модели ----------
def tfidf(**kw):
    params = dict(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2), min_df=2,
                  max_features=60000, sublinear_tf=True)
    params.update(kw)
    return TfidfVectorizer(**params)


def single_text(col="text"):
    return ColumnTransformer([("tfidf", tfidf(), col)])


def reaction_models(seed=42):
    """{имя: (конвейер, сетка гиперпараметров для GridSearchCV)}. Вход любого конвейера — to_frame()."""
    return {
        "dummy_most_frequent": (Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), {}),
        # базовые: один TF-IDF по тексту уточняющих реплик
        "nb": (Pipeline([("f", single_text()), ("m", MultinomialNB(fit_prior=False))]),
               {"m__alpha": [0.1, 0.3, 1.0]}),
        "logreg": (Pipeline([("f", single_text()),
                             ("m", LogisticRegression(class_weight="balanced", max_iter=3000))]),
                   {"m__C": [0.3, 1, 3, 10]}),
        "linear_svm": (Pipeline([("f", single_text()), ("m", LinearSVC(class_weight="balanced"))]),
                       {"m__C": [0.03, 0.1, 0.3, 1]}),
        # свои: 
        "nb_my": (Pipeline([("f", ColumnTransformer([
            ("asst_head", tfidf(), "last_asst_head"),
            ("all", tfidf(), "all_user"),])), 
            ("m", MultinomialNB(fit_prior=False))]),
               {"m__alpha": [0.1, 0.3, 1.0]}),
        "logreg_my": (Pipeline([("f", ColumnTransformer([
            ("asst_head", tfidf(), "last_asst_head"),
            ("all", tfidf(), "all_user"),])),
                             ("m", LogisticRegression(class_weight="balanced", max_iter=3000))]),
                   {"m__C": [0.3, 1, 3, 10]}),
        "linear_svm_my": (Pipeline([("f", ColumnTransformer([
            ("asst_head", tfidf(), "last_asst_head"),
            ("all", tfidf(), "all_user"),])),
              ("m", LinearSVC(class_weight="balanced"))]),
                       {"m__C": [0.03, 0.1, 0.3, 1]}),
    }


# ---------- метрики ----------
def reaction_metrics(y_true, y_pred):
    """macro-F1 — главная. macro_f1_bal — то же, но как если бы классов в тесте было поровну
    (каждому объекту вес 1/размер его класса): скрытый тест, судя по заданию, почти равномерный."""
    y_true = np.asarray(y_true) 
    counts = pd.Series(y_true).value_counts()
    w = 1.0 / pd.Series(y_true).map(counts).to_numpy()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=REACTIONS, zero_division=0)
    out = dict(macro_f1=f1_score(y_true, y_pred, average="macro", labels=REACTIONS, zero_division=0),
               macro_f1_bal=f1_score(y_true, y_pred, average="macro", labels=REACTIONS, sample_weight=w, zero_division=0),
               accuracy=accuracy_score(y_true, y_pred),
               balanced_acc=balanced_accuracy_score(y_true, y_pred))
    for i, c in enumerate(REACTIONS):
        out[f"P_{c}"], out[f"R_{c}"], out[f"F1_{c}"] = p[i], r[i], f[i]
    return out


def confusion(y_true, y_pred):
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=REACTIONS), index=REACTIONS, columns=REACTIONS)


def plot_confusions(y_true, preds, ncols=3):
    """Матрицы ошибок нескольких моделей рядом. В клетке: число объектов и доля от истинного класса
    (строки нормированы, так что провал на редком классе виден, хотя объектов в нём мало)."""
    import matplotlib.pyplot as plt
    names = list(preds)
    ncols = min(ncols, len(names))
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    k = len(REACTIONS)
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(np.asarray(y_true), np.asarray(preds[name]), labels=REACTIONS)
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.5 else "black")
        ax.set_xticks(range(k), REACTIONS, fontsize=8)
        ax.set_yticks(range(k), REACTIONS, fontsize=8)
        ax.set_xlabel("предсказано")
        ax.set_ylabel("истина")
        ax.set_title(name, fontsize=10)
        ax.grid(False)
    plt.tight_layout()
    plt.show()


# ---------- инференс (используется solution.ipynb) ----------
def init_reaction_models(path=ARTIFACTS):
    """Все сохранённые модели задачи 1: {имя: конвейер}."""
    return {f[len("reaction_"):-len(".joblib")]: joblib.load(os.path.join(path, f))
            for f in sorted(os.listdir(path)) if f.startswith("reaction_") and f.endswith(".joblib")}


def predict_reaction(model, dialogs):
    return [str(p) for p in model.predict(to_frame(dialogs))]


def save_predictions(rows, path):
    pd.DataFrame(rows, columns=["dialog_id", "task", "model", "prediction"]).to_csv(path, index=False)
