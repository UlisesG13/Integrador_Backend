import csv, time, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

BASE = "http://127.0.0.1:8000"
MOTORES = ["tfidf", "bm25", "semantico"]
CONCURRENCIAS = [1, 10, 25, 50]
PETICIONES = 200
CONSULTAS = [
    "inteligencia artificial", "seguridad en redes", "agricultura con tecnologia",
    "lenguajes de programacion", "computacion en la nube", "robotica industrial",
    "energia solar", "codigo de barras", "telefonia movil", "ciberseguridad",
]


def una_peticion(engine, i):
    q = CONSULTAS[i % len(CONSULTAS)]
    t0 = time.perf_counter()
    try:
        requests.get(f"{BASE}/search", params={"q": q, "engine": engine, "k": 5}, timeout=30)
    except Exception:
        return None
    return (time.perf_counter() - t0) * 1000


def pctl(valores, p):
    valores = sorted(valores)
    k = (len(valores) - 1) * p / 100
    f = int(k)
    return valores[f] if f + 1 >= len(valores) else valores[f] + (valores[f+1]-valores[f]) * (k - f)


def correr(engine, usuarios):
    with ThreadPoolExecutor(max_workers=usuarios) as ex:
        t0 = time.perf_counter()
        lat = [x for x in ex.map(lambda i: una_peticion(engine, i), range(PETICIONES)) if x]
        dur = time.perf_counter() - t0
    return {"motor": engine, "usuarios": usuarios, "n": len(lat),
            "p50_ms": round(pctl(lat, 50), 1), "p95_ms": round(pctl(lat, 95), 1),
            "media_ms": round(statistics.mean(lat), 1), "throughput_rps": round(len(lat) / dur, 1)}


def main():
    assert requests.get(f"{BASE}/health", timeout=10).ok, "El backend no responde en " + BASE
    filas = []
    for engine in MOTORES:
        for u in CONCURRENCIAS:
            r = correr(engine, u)
            filas.append(r)
            print(f"{engine:10} u={u:>2}  p50={r['p50_ms']:>7} ms  p95={r['p95_ms']:>7} ms  "
                  f"media={r['media_ms']:>7} ms  {r['throughput_rps']:>6} rps")
    out = Path("data/benchmark.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    print("\nGuardado", out)


if __name__ == "__main__":
    main()
