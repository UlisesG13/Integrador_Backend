import json, re, unicodedata
from pathlib import Path
import spacy

_nlp = spacy.load("es_core_news_sm")
_RE_URL = re.compile(r"https?://\S+")
_RE_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_CONSERVAR = {"no", "nunca", "sin", "ni", "muy", "poco", "nada", "tampoco"}
_STOP = set(_nlp.Defaults.stop_words) - _CONSERVAR


def _sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalizar(t):
    t = unicodedata.normalize("NFC", t.lower())
    t = _RE_URL.sub(" ", t)
    t = _RE_HTML.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def preprocesar(texto):
    out = []
    for tok in _nlp(normalizar(texto)):
        if tok.is_punct or tok.is_space or tok.like_num or tok.is_stop:
            continue
        lema = _sin_acentos(tok.lemma_.lower())
        if lema in _STOP or len(lema) <= 2:
            continue
        out.append(lema)
    return out


def cargar_corpus(ruta="data/corpus.json"):
    docs = json.loads(Path(ruta).read_text(encoding="utf-8"))
    for d in docs:
        d["texto_pleno"] = d["titulo"] + ". " + d["texto"]
        d["tokens"] = preprocesar(d["texto_pleno"])
    return docs
