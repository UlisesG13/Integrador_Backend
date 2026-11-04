import random
from locust import HttpUser, task, between

CONSULTAS = ["inteligencia artificial", "seguridad en redes", "agricultura con tecnologia",
             "lenguajes de programacion", "computacion en la nube", "robotica", "energia solar"]
MOTORES = ["tfidf", "bm25", "semantico"]


class Usuario(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def buscar(self):
        q = random.choice(CONSULTAS)
        m = random.choice(MOTORES)
        self.client.get(f"/search?q={q}&engine={m}&k=5", name=f"search:{m}")
