# handler.py — RunPod proxy -> HF Space (phenixrhyder/3D-Minecraft-Skin-Generator)
import os, io, base64, time, traceback, urllib.parse
from typing import Any, Dict, List
from PIL import Image
import requests
import runpod
from gradio_client import Client

VERSION = "space-proxy v6"

HF_SPACE_ID  = os.getenv("HF_SPACE_ID", "phenixrhyder/3D-Minecraft-Skin-Generator")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # opcional: p.ej. https://phenixrhyder-3d-minecraft-skin-generator.hf.space/
HF_TOKEN     = os.getenv("HF_TOKEN")      # MUY recomendable (evita rate limit y 403 en /file=)

_client: Client | None = None

def _space_base_url() -> str:
    if HF_SPACE_URL:
        return HF_SPACE_URL.rstrip("/")
    # construir https://owner-repo.hf.space
    return f"https://{HF_SPACE_ID.replace('/', '-').lower()}.hf.space"

def _boot():
    global _client
    base = _space_base_url()
    print(f"[boot] {VERSION} — connecting to space: {HF_SPACE_ID} (url={base})")
    _client = Client(base, hf_token=HF_TOKEN, verbose=False)
    try:
        info = _client.view_api(return_format="dict")
        print(f"[boot] view_api ok; named_endpoints={list((info or {}).get('named_endpoints', {}).keys())}")
    except Exception as e:
        print("[boot] view_api failed:", repr(e))

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def _url_for_space_file(local_path: str) -> str:
    # Convierte '/home/user/app/…png' -> 'https://<space>.hf.space/file=/home/user/app/…png'
    base = _space_base_url()
    # Importante: encode del path manteniendo las '/'.
    quoted = urllib.parse.quote(local_path, safe="/")
    return f"{base}/file={quoted}"

def _download_image(url_or_path: str) -> Image.Image:
    # Acepta tanto URL absoluta como ruta local del Space
    if not url_or_path.startswith("http"):
        url_or_path = _url_for_space_file(url_or_path)

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    headers["Referer"] = _space_base_url()

    r = requests.get(url_or_path, headers=headers, timeout=120)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def _extract_candidate_paths(obj: Any) -> List[str]:
    """
    Extrae candidatos que apunten a imágenes:
      - 'https://…/file=…'
      - '/home/user/app/…png' (ruta local del Space)
      - dicts con 'path' o 'name' que contengan rutas
    """
    hits: List[str] = []

    def add_if_path(s: str):
        if not s:
            return
        s = s.strip()
        if not s:
            return
        if s.startswith("http") and "/file=" in s:
            hits.append(s)
        elif s.startswith("/home/") or s.startswith("/tmp/") or s.endswith(".png"):
            hits.append(s)

    def walk(x: Any):
        if isinstance(x, str):
            add_if_path(x)
        elif isinstance(x, dict):
            # busca campos conocidos
            for k in ("url", "path", "name", "image", "generated_minecraft_skin_image_asset"):
                v = x.get(k)
                if isinstance(v, str):
                    add_if_path(v)
            # recorre todo
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    return hits

def _build_inputs(p: Dict[str, Any]) -> List[Any]:
    prompt = (p.get("prompt") or p.get("text") or p.get("input") or "").strip()
    if not prompt:
        raise ValueError("missing prompt")

    sd_model  = p.get("stable_diffusion_model") or "xl"
    steps     = float(p.get("steps") or p.get("num_inference_steps") or 28)
    cfg       = float(p.get("guidance_scale") or p.get("cfg") or 6.5)
    precision = p.get("model_precision_type") or "fp16"
    seed_raw  = p.get("seed", None)
    try:
        seed_val = None if seed_raw in (None, "", "random", "rnd", "auto") else float(seed_raw)
    except Exception:
        seed_val = None

    uniq = int(time.time() * 1000)
    filename = p.get("filename") or f"skin-{uniq}.png"
    model_3d = bool(p.get("model_3d", True))
    verbose  = bool(p.get("verbose", False))

    # Orden posicional exacto del Space:
    # predict(prompt, stable_diffusion_model, num_inference_steps, guidance_scale,
    #         model_precision_type, seed, filename, model_3d, verbose)
    return [prompt, sd_model, steps, cfg, precision, seed_val, filename, model_3d, verbose]

def handler(event: Dict[str, Any]):
    global _client
    try:
        if _client is None:
            _boot()

        p = (event.get("input") or {}) if isinstance(event, dict) else {}
        if p.get("warmup"):
            return {"status": "WARM", "ok": True, "version": VERSION}

        inputs = _build_inputs(p)
        prompt, sd_model, steps, cfg, precision, seed, filename, model_3d, verbose = inputs
        short_prompt = (prompt[:120] + "…") if len(prompt) > 120 else prompt
        print(f"[run] api=/predict prompt={short_prompt!r} sd={sd_model} steps={steps} cfg={cfg} "
              f"prec={precision} seed={seed} file={filename} 3d={model_3d} verbose={verbose}")

        # Evita outputs “pegados”
        try:
            _client.reset_session()
        except Exception:
            pass

        # 1) Ejecuta por cola
        job = _client.submit(*inputs, api_name="/predict")
        result = job.outputs()  # espera finalización

        # Log de forma segura (shape solo)
        def _shape(x: Any) -> str:
            if isinstance(x, dict):
                return f"dict({list(x.keys())[:6]})"
            if isinstance(x, list):
                return f"list(len={len(x)})"
            if isinstance(x, tuple):
                return f"tuple(len={len(x)})"
            return type(x).__name__
        print(f"[run] outputs shape: { _shape(result) }")

        # 2) Extrae rutas/urls de imagen
        paths = _extract_candidate_paths(result)
        print(f"[run] found {len(paths)} candidate files")

        # 3) Descarga cada imagen
        images: List[Image.Image] = []
        for pth in paths:
            try:
                images.append(_download_image(pth))
            except Exception as e:
                print(f"[download] failed {pth}: {e!r}")

        # Fallback: intenta predict directo si no encontramos nada
        if not images:
            try:
                print("[fallback] client.predict()")
                pred = _client.predict(*inputs, api_name="/predict")
                print(f"[fallback] predict shape: { _shape(pred) }")
                for pth in _extract_candidate_paths(pred):
                    try:
                        images.append(_download_image(pth))
                    except Exception as e:
                        print(f"[fallback download] {pth}: {e!r}")
            except Exception as e:
                print("[fallback] predict failed:", repr(e))

        if not images:
            return {"status": "FAILED", "error": "Space returned no downloadable images", "version": VERSION}

        images_b64 = [_to_b64(im) for im in images]
        return {
            "status": "COMPLETED",
            "images": images_b64,
            "output": {"images": images_b64, "prompt": prompt},
            "meta": {
                "count": len(images_b64),
                "sd_model": sd_model,
                "steps": steps,
                "cfg": cfg,
                "precision": precision,
                "seed": seed,
                "filename": filename,
                "version": VERSION,
            },
        }

    except Exception as e:
        print("[handler] FAILED:", repr(e))
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e), "version": VERSION}

runpod.serverless.start({"handler": handler})