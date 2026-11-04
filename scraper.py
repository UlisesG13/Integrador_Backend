import json, re, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://es.wikipedia.org/w/api.php"
UA = "IntegradorMineriaDatos/1.0 (https://github.com/UlisesG13; proyecto academico UPCh)"

CATEGORIAS = [
    "Tecnología", "Informática", "Inteligencia artificial", "Internet",
    "Telecomunicaciones", "Hardware", "Software", "Redes informáticas",
    "Lenguajes de programación", "Ciberseguridad", "Robótica",
    "Dispositivos electrónicos", "Telefonía móvil", "Computación en la nube",
]
MIN_PALABRAS = 100
OBJETIVO = 60
MAX_PALABRAS_GUARDAR = 400

sesion = requests.Session()
sesion.headers.update({"User-Agent": UA})
_retry = Retry(total=6, backoff_factor=2.0, status_forcelist=[429, 500, 502, 503],
               respect_retry_after_header=True, allowed_methods=["GET"])
sesion.mount("https://", HTTPAdapter(max_retries=_retry))


def get(params):
    params = {**params, "format": "json", "maxlag": 5}
    r = sesion.get(API, params=params, timeout=40)
    r.raise_for_status()
    return r.json()


def titulos_de_categoria(cat, limite=40):
    data = get({"action": "query", "list": "categorymembers",
                "cmtitle": f"Categoría:{cat}", "cmtype": "page", "cmlimit": limite})
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def extracto(titulo):
    data = get({"action": "query", "prop": "extracts", "explaintext": 1,
                "exsectionformat": "plain", "titles": titulo, "redirects": 1})
    for _, p in data.get("query", {}).get("pages", {}).items():
        return p.get("extract", "")
    return ""


def limpiar(texto):
    texto = re.sub(r"=+\s*[^=\n]+\s*=+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def main():
    candidatos, vistos = [], set()
    for cat in CATEGORIAS:
        try:
            for t in titulos_de_categoria(cat):
                if t not in vistos:
                    vistos.add(t); candidatos.append((t, cat))
        except Exception as e:
            print(f"  (categoría {cat}: {e})")
        time.sleep(1.0)
    print(f"Títulos candidatos reunidos: {len(candidatos)}")

    corpus = []
    for t, cat in candidatos:
        if len(corpus) >= OBJETIVO:
            break
        try:
            txt = limpiar(extracto(t))
        except Exception:
            continue
        palabras = txt.split()
        if len(palabras) < MIN_PALABRAS:
            continue
        txt = " ".join(palabras[:MAX_PALABRAS_GUARDAR])
        corpus.append({"id": f"d{len(corpus)+1:03d}", "titulo": t, "texto": txt,
                       "url": "https://es.wikipedia.org/wiki/" + t.replace(" ", "_"),
                       "palabras": len(txt.split()), "categoria": cat})
        print(f"  [{corpus[-1]['id']}] {t} ({corpus[-1]['palabras']} palabras)")
        time.sleep(0.8)

    out = Path("data/corpus.json"); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardados {len(corpus)} documentos en {out}")
    if corpus:
        ns = [d["palabras"] for d in corpus]
        print(f"Palabras: min={min(ns)}, media={sum(ns)//len(ns)}, max={max(ns)}")


if __name__ == "__main__":
    main()
