import math
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import preprocesar


class MotorTFIDF:
    nombre = "TF-IDF"

    def __init__(self, docs):
        self.ids = [d["id"] for d in docs]
        self.tokens = {d["id"]: d["tokens"] for d in docs}
        N = len(self.ids)
        self.df = Counter(t for i in self.ids for t in set(self.tokens[i]))
        self.idf = {t: math.log(N / self.df[t]) for t in self.df}
        self.index = {i: self._vec(self.tokens[i]) for i in self.ids}

    def _tf(self, toks):
        n = len(toks)
        return {t: f / n for t, f in Counter(toks).items()} if n else {}

    def _vec(self, toks):
        return {t: w * self.idf.get(t, 0.0) for t, w in self._tf(toks).items()}

    @staticmethod
    def _cos(a, b):
        num = sum(a[t] * b[t] for t in set(a) & set(b))
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def buscar(self, consulta, k=5):
        vq = self._vec(preprocesar(consulta))
        puntajes = [(i, self._cos(vq, self.index[i])) for i in self.ids]
        puntajes.sort(key=lambda x: (-x[1], x[0]))
        return [(i, s) for i, s in puntajes[:k] if s > 0]


class MotorBM25:
    nombre = "BM25"

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.ids = [d["id"] for d in docs]
        self.tokens = {d["id"]: d["tokens"] for d in docs}
        N = len(self.ids)
        self.df = Counter(t for i in self.ids for t in set(self.tokens[i]))
        self.idf = {t: math.log(1 + (N - self.df[t] + 0.5) / (self.df[t] + 0.5)) for t in self.df}
        self.tfd = {i: Counter(self.tokens[i]) for i in self.ids}
        self.long = {i: len(self.tokens[i]) for i in self.ids}
        self.avgdl = sum(self.long.values()) / N

    def _termino(self, t, i):
        f = self.tfd[i].get(t, 0)
        if f == 0:
            return 0.0
        dl = self.long[i]
        return self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

    def buscar(self, consulta, k=5):
        qt = [t for t in preprocesar(consulta) if t in self.df]
        puntajes = [(i, sum(self._termino(t, i) for t in qt)) for i in self.ids]
        puntajes.sort(key=lambda x: (-x[1], x[0]))
        return [(i, s) for i, s in puntajes[:k] if s > 0]


class MotorSemantico:
    nombre = "Semántico"

    def __init__(self, docs, modelo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.ids = [d["id"] for d in docs]
        self.bi = SentenceTransformer(modelo)
        emb = self.bi.encode([d["texto_pleno"] for d in docs],
                             normalize_embeddings=True, convert_to_numpy=True)
        self.emb = dict(zip(self.ids, emb))

    def buscar(self, consulta, k=5):
        vq = self.bi.encode([consulta], normalize_embeddings=True, convert_to_numpy=True)[0]
        puntajes = [(i, float(self.emb[i] @ vq)) for i in self.ids]
        puntajes.sort(key=lambda x: (-x[1], x[0]))
        return puntajes[:k]
