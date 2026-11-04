import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .corpus import cargar_corpus
from .engines import MotorTFIDF, MotorBM25, MotorSemantico

app = FastAPI(title="Integrador — Motores de Búsqueda", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DOCS = cargar_corpus("data/corpus.json")
META = {d["id"]: d for d in DOCS}
MOTORES = {"tfidf": MotorTFIDF(DOCS), "bm25": MotorBM25(DOCS), "semantico": MotorSemantico(DOCS)}


def _formatea(resultados):
    salida = []
    for doc_id, score in resultados:
        d = META[doc_id]
        salida.append({"id": doc_id, "titulo": d["titulo"], "score": round(float(score), 4),
                       "url": d["url"], "snippet": d["texto"][:200] + "…"})
    return salida


@app.get("/health")
def health():
    return {"ok": True, "documentos": len(DOCS), "motores": list(MOTORES)}


@app.get("/stats")
def stats():
    ns = [d["palabras"] for d in DOCS]
    return {"documentos": len(DOCS), "palabras_min": min(ns), "palabras_media": sum(ns) // len(ns),
            "palabras_max": max(ns), "fuente": "Wikipedia en español (tema: tecnología)"}


@app.get("/search")
def search(q: str = Query(..., min_length=1), engine: str = "bm25", k: int = 5):
    engine = engine.lower()
    if engine not in MOTORES:
        return {"error": f"motor inválido; use uno de {list(MOTORES)}"}
    t0 = time.perf_counter()
    resultados = MOTORES[engine].buscar(q, k=k)
    latency = (time.perf_counter() - t0) * 1000
    return {"query": q, "engine": engine, "latency_ms": round(latency, 2),
            "resultados": _formatea(resultados)}


@app.get("/compare")
def compare(q: str = Query(..., min_length=1), k: int = 5):
    out = {}
    for clave, motor in MOTORES.items():
        t0 = time.perf_counter()
        resultados = motor.buscar(q, k=k)
        out[clave] = {"nombre": motor.nombre, "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                      "resultados": _formatea(resultados)}
    return {"query": q, "motores": out}
