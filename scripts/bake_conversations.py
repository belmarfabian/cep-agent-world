"""Genera conversaciones reales del 'modelo mundo' y las hornea en el dashboard.

Cada agente es un encuestado real de la CEP N.95: su edad, sexo, region, nivel
socioeconomico y posicion politica se convierten en un system prompt que un
modelo de lenguaje encarna. Dos agentes conectados conversan en espanol chileno
y al final reportan su opinion; el resultado se guarda en
`data/processed/conversations.js` (const CHATS), que el dashboard reproduce.

Proveedores (elige con --provider; por defecto Groq, que tiene capa gratuita):
  groq      GROQ_API_KEY      -> https://console.groq.com (gratis, sin tarjeta)
  gemini    GEMINI_API_KEY    -> https://aistudio.google.com (capa gratuita)
  anthropic ANTHROPIC_API_KEY -> de pago, muy barato con Haiku

Uso:
  python scripts/bake_conversations.py                  # Groq, modelo gratuito
  python scripts/bake_conversations.py --provider gemini
  python scripts/bake_conversations.py --provider anthropic --model claude-haiku-4-5

El conjunto de pares (uno por tema) esta fijado para que el mapa resalte agentes
reales; edita PAIRS para cambiarlo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "data" / "processed" / "simulacion_interactiva.html"
OUT = ROOT / "data" / "processed" / "conversations.js"

# (agente_izquierda, agente_derecha, indice_de_tema en labs[])
PAIRS = [
    (1095, 1097, 0), (1137, 1140, 1), (1116, 1118, 2), (1197, 1198, 3),
    (1036, 1039, 4), (1033, 1031, 5), (1018, 1016, 6),
]

TOPICS = [
    "Deberian eliminarse las AFP y volver a un sistema de reparto?",
    "Hay que aplicar mano dura contra la delincuencia o priorizar la prevencion?",
    "Deberia restringirse la inmigracion en Chile?",
    "Como evalua al gobierno actual?",
    "Debe el Estado intervenir para reducir la desigualdad?",
    "La educacion deberia ser gratuita para todos?",
    "Que se debe priorizar: el medioambiente o el crecimiento economico?",
]

PROVIDERS = {
    "groq": dict(env="GROQ_API_KEY", model="llama-3.3-70b-versatile",
                 url="https://api.groq.com/openai/v1/chat/completions", kind="openai"),
    "gemini": dict(env="GEMINI_API_KEY", model="gemini-2.0-flash",
                   url="https://generativelanguage.googleapis.com/v1beta/models", kind="gemini"),
    "anthropic": dict(env="ANTHROPIC_API_KEY", model="claude-haiku-4-5",
                      url="https://api.anthropic.com/v1/messages", kind="anthropic"),
}


def load_agents() -> dict[int, dict]:
    html = HTML.read_text(encoding="utf-8")
    d = json.loads(re.search(r"const D=(\{.*?\});", html, re.S).group(1))
    return {a["i"]: a for a in d["a"]}


def system_prompt(a: dict) -> str:
    pos = a["b"]
    if pos <= -0.5:
        pol = "Te identificas con la izquierda."
    elif pos < -0.1:
        pol = "Te identificas con la centro-izquierda."
    elif pos <= 0.1:
        pol = "Te ubicas en el centro politico."
    elif pos < 0.5:
        pol = "Te identificas con la centro-derecha."
    else:
        pol = "Te identificas con la derecha."
    registro = ("Usa un lenguaje coloquial y directo, con expresiones populares chilenas."
                if a["g"] in ("D", "E") else
                "Usa un lenguaje formal pero cercano." if a["g"] in ("ABC1", "C2") else
                "Usa un lenguaje natural y cotidiano chileno.")
    return (
        f"Eres un/a chileno/a de {a['a']} anos, {a['s'].lower()}, que vive en la region "
        f"de {a['r']}. Tu nivel socioeconomico es {a['g']}. {pol}\n"
        "INSTRUCCIONES:\n"
        f"- Responde SIEMPRE en espanol chileno. {registro}\n"
        "- Manten consistencia con tu perfil. No cambies de opinion facilmente, "
        "pero puedes matizar ante argumentos solidos.\n"
        "- Tus respuestas deben ser breves: 2 a 3 oraciones por turno."
    )


def post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(prov: dict, key: str, model: str, system: str, msgs: list[dict]) -> str:
    """msgs: [{'role':'user'|'assistant','content':str}]. Devuelve el texto."""
    if prov["kind"] == "openai":
        body = {"model": model, "messages": [{"role": "system", "content": system}, *msgs],
                "max_tokens": 200, "temperature": 0.8}
        r = post(prov["url"], body, {"Authorization": f"Bearer {key}"})
        return r["choices"][0]["message"]["content"].strip()
    if prov["kind"] == "anthropic":
        body = {"model": model, "system": system, "messages": msgs,
                "max_tokens": 200, "temperature": 0.8}
        r = post(prov["url"], body, {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return r["content"][0]["text"].strip()
    if prov["kind"] == "gemini":
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in msgs]
        body = {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.8}}
        url = f"{prov['url']}/{model}:generateContent?key={key}"
        r = post(url, body, {})
        return r["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise ValueError(prov["kind"])


def opinion(prov, key, model, system, history, topic) -> float:
    q = (f"Sobre la pregunta: '{topic}'. Responde SOLO con un numero entre -1 y 1, "
         "donde -1 es totalmente la postura de izquierda y 1 la de derecha. Solo el numero.")
    txt = chat(prov, key, model, system, [*history, {"role": "user", "content": q}])
    m = re.search(r"-?\d*\.?\d+", txt)
    if not m:
        return 0.0
    return max(-1.0, min(1.0, float(m.group())))


def run_pair(prov, key, model, agents, ai, bi, ti) -> dict:
    a, b = agents[ai], agents[bi]
    sa, sb = system_prompt(a), system_prompt(b)
    topic = TOPICS[ti]
    before = [a["b"], b["b"]]

    hist_a = [{"role": "user", "content": f"Alguien te pregunta: {topic} Da tu opinion."}]
    turns = []
    ra = chat(prov, key, model, sa, hist_a)
    hist_a.append({"role": "assistant", "content": ra})
    turns.append({"s": 0, "t": ra})

    speaker, sys_s, hist_s = "b", sb, [{"role": "user", "content": ra}]
    last = ra
    for _ in range(3):
        resp = chat(prov, key, model, sys_s, hist_s)
        hist_s.append({"role": "assistant", "content": resp})
        turns.append({"s": 1 if speaker == "b" else 0, "t": resp})
        last = resp
        if speaker == "b":
            speaker, sys_s, hist_s = "a", sa, [*hist_a, {"role": "user", "content": resp}]
        else:
            speaker, sys_s, hist_s = "b", sb, [*hist_s, {"role": "user", "content": last}]

    after = [
        opinion(prov, key, model, sa, hist_a, topic),
        opinion(prov, key, model, sb, hist_s, topic),
    ]
    print(f"  {ai}<->{bi} tema {ti}: op {before} -> {after}")
    return {"ai": ai, "bi": bi, "topic": ti, "before": before, "after": after, "turns": turns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default="groq")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    prov = PROVIDERS[args.provider]
    model = args.model or prov["model"]
    key = os.environ.get(prov["env"])
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(prov["env"] + "="):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit(f"Falta {prov['env']}. Consíguela gratis y expórtala, o ponla en .env")

    agents = load_agents()
    print(f"Generando {len(PAIRS)} conversaciones con {args.provider} ({model})...")
    chats = [run_pair(prov, key, model, agents, ai, bi, ti) for ai, bi, ti in PAIRS]

    js = ("// Generado por scripts/bake_conversations.py — conversaciones reales.\n"
          f'const CHATS_META={{demo:false,modelo:"{args.provider}:{model}"}};\n'
          "const CHATS=" + json.dumps(chats, ensure_ascii=False) + ";\n")
    OUT.write_text(js, encoding="utf-8")
    print(f"Escrito {OUT} ({len(js)/1024:.1f} KB). Recarga el dashboard.")


if __name__ == "__main__":
    main()
