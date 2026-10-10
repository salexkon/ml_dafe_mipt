"""HW1, задача 2 (темы диалогов). Самостоятельный файл утилит: чтение, очистка текста,
признаки, модели тем, метрики кластеризации, инференс. Не зависит от solution_utils.py.
Инструменты — с семинаров 1–5: TF-IDF, стемминг nltk, KMeans, SVD, NB, LogReg, SVM, GridSearchCV, Pipeline + joblib."""
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")


def read_dialogs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- обработка текста ----------
def clean_text(text):
    """Чистка одной реплики. Самодостаточная функция: всё нужное лежит внутри.
    Порядок: служебные символы -> ссылки -> код и математика (построчно) -> косые черты и непечатные символы ->
    пробелы -> обрезка длинного с двух концов. Вместо вырезанного дописываем маркеры zzurl, zzcode, zzmath."""
    max_chars, head_chars, min_len = 2000, 1200, 200
    symbols = set("{}[]()<>=;_#$\\|^*+-&%~`")                 # «/» не берём: он есть в ссылках и обычном тексте
    code_starts = ("def ", "class ", "import ", "#include", "return ", "function ", "const ", "public ",
                   "private ", "void ", "package ", "using namespace", "select ", "<?php", "$ ", ">>> ")
    latex_marks = ("\\frac", "\\sum", "\\int", "\\begin", "\\end", "\\left", "\\cdot", "\\sqrt", "\\lim",
                   "$$", "\\(", "\\[", "\\times", "\\alpha", "\\beta")
    latex_cmds = ("\\neq", "\\nabla", "\\nu", "\\not", "\\theta", "\\tau", "\\times", "\\tan", "\\to", "\\rho", "\\right")

    def symbol_share(s):
        s = "".join(s.split())
        return sum(ch in symbols for ch in s) / len(s) if s else 0.0

    def looks_like_math(line):
        s = line.strip()
        if not s:
            return False
        if any(mark in s for mark in latex_marks):
            return True
        chars = "".join(s.split())
        numeric = sum(ch.isdigit() or ch in "+-*/=^<>().,%" for ch in chars) / len(chars)
        long_words = sum(w.isalpha() and len(w) >= 4 for w in s.split())
        return numeric >= 0.5 and long_words <= 2

    def looks_like_code(line):
        s = line.strip()
        if not s:
            return False
        sym = symbol_share(s)
        if s.endswith((";", "{", "}")) and sym > 0.04:
            return True
        if s.lower().startswith(code_starts) and (sym > 0.02 or s.endswith(":") or line[0].isspace()):
            return True
        if line.startswith(("    ", "\t")) and sym > 0.05:
            return True
        return sym > 0.18

    def is_url(word):
        word = word.lower().lstrip("([{<\"'")
        return "://" in word or word.startswith("www.")

    # 1. буквальные «\n», «\r», «\t» -> настоящие переводы строк и пробелы (команды LaTeX на «\n», «\t», «\r» прячем)
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    shield = any(mark in text for mark in latex_marks)
    if shield:
        for cmd in latex_cmds:
            text = text.replace(cmd, "\u0001" + cmd[1:])
    for esc, real in (("\\r\\n", "\n"), ("\\n", "\n"), ("\\r", "\n"), ("\\t", " ")):
        text = text.replace(esc, real)
    if shield:
        text = text.replace("\u0001", "\\")

    # 2. ссылки: до поиска кода, в них много «=», «&», «%»
    had_url, lines = False, []
    for line in text.split("\n"):
        words = line.split(" ")
        kept_words = [w for w in words if not is_url(w)]
        had_url = had_url or len(kept_words) < len(words)
        lines.append(" ".join(kept_words))
    text = "\n".join(lines)

    # 3. код и математика, построчно (только в длинных репликах: короткое «print(x)» в чате не трогаем)
    had_code = had_math = False
    if len(text) >= min_len:
        kept_lines, in_fence = [], False
        for line in text.split("\n"):
            if line.strip().startswith("```"):                # блок кода в тройных кавычках
                in_fence, had_code = not in_fence, True
            elif in_fence:
                had_code = True
            elif looks_like_math(line):                       # раньше кода: «2x^2 + 3x = 0» по символам похоже на код
                had_math = True
            elif looks_like_code(line):
                had_code = True
            else:
                kept_lines.append(line)
        text = "\n".join(kept_lines)

    # 4. косые черты подряд -> одна; непечатные символы -> пробел; пробелы склеиваем
    collapsed = []
    for ch in text:
        if ch in "/\\" and collapsed and collapsed[-1] in "/\\":
            continue
        collapsed.append(ch if ch.isprintable() or ch.isspace() else " ")
    text = " ".join("".join(collapsed).split())

    # 5. слишком длинное: начало (постановка) и конец (сама просьба)
    if len(text) > max_chars:
        text = text[:head_chars] + " " + text[-(max_chars - head_chars):]
    markers = (" zzcode" if had_code else "") + (" zzmath" if had_math else "") + (" zzurl" if had_url else "")
    return (text + markers).strip()


def normalize(text):
    """Токены wordpunct -> нижний регистр -> основы слов: русским стеммером для кириллицы, английским для латиницы.
    Слова других письменностей (китайский и т. д.) wordpunct не режет, целая фраза — один токен: заменяем маркером
    zzother. Стоп-слова НЕ убираем: в списке есть «not» и «no», а для реакции отрицание важно (семинар 4).
    Знаки «!» и «?» оставляем: они несут эмоцию, а стандартный token_pattern sklearn их бы выбросил."""
    cache = normalize.__dict__.setdefault("cache", {})          # одно и то же слово стеммим один раз
    stemmers = normalize.__dict__.setdefault(
        "stemmers", {"lat": SnowballStemmer("english"), "ru": SnowballStemmer("russian")})
    out = []
    for tok in wordpunct_tokenize(clean_text(text).lower()):
        if tok.isalpha():
            first = next(ch for ch in tok if ch.isalpha())
            if "Ѐ" <= first <= "ӿ":
                script = "ru"
            elif first < "ɏ":
                script = "lat"
            else:
                out.append("zzother")
                continue
            if tok not in cache:
                cache[tok] = stemmers[script].stem(tok)
            out.append(cache[tok])
        elif tok in ("!", "?"):
            out.append(tok)
    return " ".join(out)


# ===================== ЗАДАЧА 2: ТЕМЫ  =====================
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.metrics import normalized_mutual_info_score, v_measure_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import KNeighborsClassifier

TOPIC_HEAD_WORDS = 40                  # сколько первых слов первого ответа модели берём


def to_topic_frame(dialogs):
    """Диалоги -> таблица признаков для тем. Тему задаёт то, о чём просит пользователь, поэтому берём
    первую реплику пользователя, остальные его реплики и начало первого ответа модели.
    Каждая реплика чистится отдельно (иначе лимит длины съедается одной вставкой)."""
    rows = []
    for d in dialogs:
        turns = d["turns"]
        us = [t["text"] or "" for t in turns if t["role"] == "user"]
        asst = [t["text"] or "" for t in turns if t["role"] == "assistant"]
        labels = d.get("labels") or {}
        rows.append(dict(
            dialog_id=d["dialog_id"],
            first_user=normalize(us[0]) if us else "",
            rest_user=" ".join(normalize(u) for u in us[1:]),
            first_asst=normalize(" ".join(asst[0].split()[:TOPIC_HEAD_WORDS])) if asst else "",
            raw_first=clean_text(us[0]) if us else "",      # очищенный, но не токенизированный: показывать примеры
            n_user=len(us),
            group="".join(ch for ch in us[0].lower() if ch.isalnum())[:80] if us else d["dialog_id"],
            task_type=labels.get("task_type"),               # только для сверки, темами не являются
            reaction=labels.get("reaction"),
        ))
    return pd.DataFrame(rows)


def topic_tfidf(**kw):
    """TF-IDF для тем. max_df=0.4 выбрасывает слова, которые есть почти в каждом диалоге (служебные),
    min_df=5 — опечатки и уникальные слова."""
    params = dict(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2), min_df=5, max_df=0.4,
                  max_features=50000, sublinear_tf=True)
    params.update(kw)
    return TfidfVectorizer(**params)


def topic_features(weights=(1.0, 0.5, 0.5)):
    """Три словаря: первая реплика пользователя (главная), остальные реплики и начало ответа модели."""
    return ColumnTransformer(
        [("first", topic_tfidf(), "first_user"), ("rest", topic_tfidf(), "rest_user"),
         ("asst", topic_tfidf(), "first_asst")],
        transformer_weights=dict(zip(["first", "rest", "asst"], weights)))


def topic_lsa(n_components=100, seed=42):
    """TF-IDF -> SVD (семинар 5) -> нормировка длины: в этом пространстве кластеризуем."""
    return make_pipeline(topic_features(), TruncatedSVD(n_components, random_state=seed), Normalizer(copy=False))


def topic_models(seed=42):
    """{имя: (конвейер, сетка гиперпараметров)}. Вход любого конвейера — to_topic_frame()."""
    return {
        "dummy_one_topic": (Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), {}),
        "dummy_random_topics": (Pipeline([("m", DummyClassifier(strategy="uniform", random_state=seed))]), {}),
        "nb": (Pipeline([("f", topic_features()), ("m", MultinomialNB(fit_prior=False))]),
               {"m__alpha": [0.03, 0.1, 0.3]}),
        "logreg": (Pipeline([("f", topic_features()),
                             ("m", LogisticRegression(class_weight="balanced", max_iter=2000))]),
                   {"m__C": [1, 10, 30]}),
        "linear_svm": (Pipeline([("f", topic_features()), ("m", LinearSVC(class_weight="balanced"))]),
                       {"m__C": [0.1, 0.3, 1]}),
        # своя 1: SVD (семинар 5) и ближайшие соседы (семинар 1) по косинусной близости
        "own_lsa_knn": (Pipeline([("f", topic_features()), ("svd", TruncatedSVD(100, random_state=seed)),
                                  ("norm", Normalizer()),
                                  ("m", KNeighborsClassifier(weights="distance"))]),
                        {"m__n_neighbors": [5, 15, 30]}),
        # своя 2: SVD и SVM с RBF-ядром (семинар 5)
        "own_lsa_rbf_svm": (Pipeline([("f", topic_features()), ("svd", TruncatedSVD(100, random_state=seed)),
                                      ("norm", Normalizer()),
                                      ("m", SVC(kernel="rbf", class_weight="balanced"))]),
                            {"m__C": [1, 10]}),
    }


# ---------- метрики для тем: сравниваем разбиения, названия тем не важны ----------
def purity_score(y_ref, y_pred):
    """Для каждой вашей темы берём самую частую эталонную; чистота — доля объектов, попавших в «свою» тему."""
    cm = contingency_matrix(y_ref, y_pred)
    return cm.max(axis=0).sum() / cm.sum()


def partition_scores(y_ref, y_pred):
    return {"ARI": adjusted_rand_score(y_ref, y_pred), "NMI": normalized_mutual_info_score(y_ref, y_pred),
            "однородность": homogeneity_score(y_ref, y_pred), "полнота": completeness_score(y_ref, y_pred),
            "V-мера": v_measure_score(y_ref, y_pred), "чистота": purity_score(y_ref, y_pred)}


def topic_metrics(y_ref, y_pred):
    return {"accuracy": accuracy_score(y_ref, y_pred),
            "macro_f1": f1_score(y_ref, y_pred, average="macro", zero_division=0),
            **partition_scores(y_ref, y_pred)}


def metrics_by_group(df, group_col, true_col, pred_col, min_n=1):
    """Метрики классификации отдельно в каждой группе (например, реакция по темам). Макро-F1 считается
    только по классам, которые в группе есть; в маленьких группах числа шумят, смотрите на n."""
    rows = []
    for g, part in df.groupby(group_col):
        yt, yp = part[true_col].to_numpy(), part[pred_col].to_numpy()
        if len(part) < min_n:
            continue
        present = sorted(set(yt) | set(yp))
        rows.append({group_col: g, "n": len(part), "доля": len(part) / len(df),
                     "accuracy": accuracy_score(yt, yp),
                     "macro_f1": f1_score(yt, yp, labels=present, average="macro", zero_division=0),
                     "доля negative": (yt == "negative").mean(),
                     "recall negative": (yp[yt == "negative"] == "negative").mean() if (yt == "negative").any() else np.nan})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ---------- инференс (используется solution.ipynb) ----------
def init_topic_models(path=ARTIFACTS):
    """Все сохранённые модели задачи 2: {имя: конвейер}."""
    return {f[len("topic_"):-len(".joblib")]: joblib.load(os.path.join(path, f))
            for f in sorted(os.listdir(path)) if f.startswith("topic_") and f.endswith(".joblib")}


def predict_topic(model, dialogs):
    return [str(p) for p in model.predict(to_topic_frame(dialogs))]
