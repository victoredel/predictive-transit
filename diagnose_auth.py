"""
diagnose_auth.py — Script de diagnóstico para autenticación SPTrans Olho Vivo
==============================================================================
Ejecutar localmente (fuera de Docker) para depurar el endpoint de auth:
    pip install requests
    python diagnose_auth.py

Prueba 6 variantes distintas del request para identificar cuál acepta el servidor.
"""

import os
import sys

try:
    import requests
except ImportError:
    print("Instala requests: pip install requests")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("SPTRANS_TOKEN", "9717b92ee49598ee19095c425b8424ee8d0c11066de2e00b85112cce26db649c").strip()
BASE = "https://api.olhovivo.sptrans.com.br/v2.1"
AUTH_URL = f"{BASE}/Login/Autenticar"

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

print(f"\n{'='*60}")
print(f"DIAGNÓSTICO SPTrans Olho Vivo Auth")
print(f"Token: {TOKEN[:16]}...{TOKEN[-8:]}")
print(f"{'='*60}\n")

session = requests.Session()
session.headers.update({
    "User-Agent": CHROME_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Origin": "https://www.sptrans.com.br",
    "Referer": "https://www.sptrans.com.br/",
})

variantes = [
    {
        "nombre": "1. POST token en query string, body vacío, CL:0",
        "method": "POST",
        "url": f"{AUTH_URL}?token={TOKEN}",
        "headers": {"Content-Length": "0", "Content-Type": "application/x-www-form-urlencoded"},
        "data": b"",
    },
    {
        "nombre": "2. POST token en body form-urlencoded",
        "method": "POST",
        "url": AUTH_URL,
        "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        "data": f"token={TOKEN}",
    },
    {
        "nombre": "3. POST token en body JSON",
        "method": "POST",
        "url": AUTH_URL,
        "headers": {"Content-Type": "application/json"},
        "data": f'{{"token":"{TOKEN}"}}',
    },
    {
        "nombre": "4. GET token en query string",
        "method": "GET",
        "url": f"{AUTH_URL}?token={TOKEN}",
        "headers": {},
        "data": None,
    },
    {
        "nombre": "5. POST query string + NO Content-Type",
        "method": "POST",
        "url": f"{AUTH_URL}?token={TOKEN}",
        "headers": {"Content-Length": "0"},
        "data": b"",
    },
    {
        "nombre": "6. POST con pre-GET de la home (warm-up de sesión)",
        "method": "POST",
        "url": f"{AUTH_URL}?token={TOKEN}",
        "headers": {"Content-Length": "0"},
        "data": b"",
        "pre_get": f"{BASE}/Linha/BuscarLinhas?termosBusca=&sentido=1",
    },
]

for v in variantes:
    print(f"\n{'─'*50}")
    print(f"▶ {v['nombre']}")

    try:
        # Pre-GET de warm-up si aplica
        if v.get("pre_get"):
            try:
                session.get(v["pre_get"], timeout=5)
                print(f"  (pre-GET ejecutado)")
            except Exception:
                pass

        resp = session.request(
            method=v["method"],
            url=v["url"],
            headers=v.get("headers", {}),
            data=v.get("data"),
            timeout=10,
            allow_redirects=True,
        )

        print(f"  Status : {resp.status_code}")
        print(f"  Body   : {resp.text[:100]!r}")

        # Mostrar cookies relevantes
        cookies_recibidas = dict(resp.cookies)
        if cookies_recibidas:
            print(f"  Cookies: {cookies_recibidas}")
        else:
            print(f"  Cookies: (ninguna)")

        # Si autenticó, probar el endpoint de posiciones
        if resp.text.strip().lower() == "true":
            print(f"\n  ✅ ¡AUTENTICACIÓN EXITOSA con variante {v['nombre'][:2]}!")
            pos_resp = session.get(
                f"{BASE}/Posicao",
                timeout=10,
            )
            body_preview = pos_resp.text[:200]
            print(f"  /Posicao status: {pos_resp.status_code}")
            print(f"  /Posicao body  : {body_preview!r}")
            break

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("Diagnóstico completado.")
print("Si todas las variantes devuelven 'false', el problema es el token en el servidor.")
print(f"{'='*60}\n")
