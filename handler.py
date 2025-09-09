# handler.py — RunPod -> HF Space proxy (phenixrhyder/3D-Minecraft-Skin-Generator)
import os, io, time, base64, traceback, urllib.parse
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import requests
import runpod
from gradio_client import Client

VERSION = "space-proxy v7"

HF_SPACE_ID  = os.getenv("HF_SPACE_ID", "phenixrhyder/3D-Minecraft-Skin-Generator")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # opcional, p.ej. https://phenixrhyder-3d-minecraft-skin-generator.hf.space/
HF_TOKEN     = os.getenv("HF_TOKEN")      # MUY recomendable (evita rate limit y 403 en /file=)

_client: Optional[Client] = None

# ---------------- util ---------------

def _space_base_url() -> str:
    if HF_SPACE_URL:
        return HF_SPACE_URL.rstrip("/")
    # https://owner-repo.hf.space
    return f"https://{HF_SPACE_ID.replace('/', '-').lower()}.hf.space"

def _boot():
    global _client
    base = _space_base_url()
    print(f"[boot] {VERSION} connect -> {HF_SPACE_ID} ({base})")
    _client = Client(base, hf_token=HF_TOKEN, verbose=False)
    try:
        info = _client.view_api(return_format="dict")  # opcional, solo logging
        print(f"[boot] view_api ok; endpoints={list((info or {}).get('named_endpoints', {}).keys())}")
    except Exception as e:
        print("[boot] view_api failed:", repr(e))

def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def _is_data_url(s: str) -> bool:
    return s.startswith("data:image/") and ";base64," in s

def _data_url_to_image(s: str) -> Image.Image:
    b64 = s.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

def _url_for_space_file(local_path: str) -> str:
    base = _space_base_url()
    quoted = urllib.parse.quote(local_path, safe="/")
    return f"{base}/file={quoted}"

def _download_image(url_or_path: str) -> Image.Image:
    url = url_or_path
    if not url.startswith("http"):
        url = _url_for_space_file(url_or_path)

    headers = {"Referer": _space_base_url()}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def _shape(x: Any) -> str:
    try:
        if isinstance(x, dict):  return f"dict({list(x.keys())[:6]})"
        if isinstance(x, list):  return f"list(len={len(x)})"
        if isinstance(x, tuple): return f"tuple(len={len(x)})"
        return type(x).__name__
    except Exception:
        return type(x).__name__

# --------- extracción de imágenes en cualquier formato ---------

def _collect_candidate_paths(obj: Any) -> List[str]:
    """Encuentra strings que parezcan rutas/urls de imagen del Space."""
    hits: List[str] = []

    def add_if(s: Optional[str]):
        if not s: return
        s = s.strip()
        if not s: return
        # URL /file= del Space
        if s.startswith("http") and "/file=" in s:
            hits.append(s); return
        # rutas locales o nombres con .png
        if s.startswith(("/home/", "/tmp/")) or s.lower().endswith(".png"):
            hits.append(s); return

    def walk(x: Any):
        if isinstance(x, str):
            add_if(x)
        elif isinstance(x, dict):
            # campos típicos
            for k in ("url", "path", "name", "image", "generated_minecraft_skin_image_asset"):
                v = x.get(k)
                if isinstance(v, str): add_if(v)
            # recursivo
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)

    walk(obj)
    return hits

def _collect_inline_images(obj: Any) -> List[Image.Image]:
    """Busca data-URL/base64 o bytes embebidos y los convierte a PIL."""
    images: List[Image.Image] = []

    def try_add(x: Any):
        try:
            if isinstance(x, (bytes, bytearray)):
                images.append(Image.open(io.BytesIO(x)).convert("RGBA")); return True
            if isinstance(x, str) and _is_data_url(x):
                images.append(_data_url_to_image(x)); return True
        except Exception:
            return False
        return False

    def walk(y: Any):
        if try_add(y): return
        if isinstance(y, dict):
            # data embebida a veces va en 'data' o 'image'
            for k in ("data", "image"):
                if k in y: try_add(y[k])
            for v in y.values(): walk(v)
        elif isinstance(y, (list, tuple)):
            for v in y: walk(v)

    walk(obj)
    return images

# --------- construcción de inputs (orden posicional de /predict) ---------

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

    # evita colisiones de caché del space
    uniq = int(time.time() * 1000)
    filename = p.get("filename") or f"skin-{uniq}.png"
    model_3d = bool(p.get("model_3d", True))
    verbose  = bool(p.get("verbose", False))

    # Orden exacto del Space:
    # predict(prompt, stable_diffusion_model, num_inference_steps, guidance_scale,
    #         model_precision_type, seed, filename, model_3d, verbose)
    return [prompt, sd_model, steps, cfg, precision, seed_val, filename, model_3d, verbose]

# ---------------- handler ----------------

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

        # Evita “state pegado” en la sesión
        try:
            _client.reset_session()
        except Exception:
            pass

        # 1) Submit por cola (fiable para Spaces con queue)
        job = _client.submit(*inputs, api_name="/predict")
        result = job.outputs()
        print(f"[run] outputs shape: { _shape(result) }")

        # 2) Intenta inline (data-url/bytes) primero
        images = _collect_inline_images(result)

        # 3) Luego rutas/urls y descarga
        if not images:
            paths = _collect_candidate_paths(result)
            print(f"[run] candidate file paths: {len(paths)}")
            for pth in paths:
                try:
                    images.append(_download_image(pth))
                except Exception as e:
                    print(f"[download] {pth}: {e!r}")

        # 4) Fallback a predict() si aún no hay nada
        if not images:
            print("[fallback] direct predict()")
            pred = _client.predict(*inputs, api_name="/predict")
            print(f"[fallback] shape: { _shape(pred) }")
            images = _collect_inline_images(pred)
            if not images:
                paths = _collect_candidate_paths(pred)
                print(f"[fallback] candidate file paths: {len(paths)}")
                for pth in paths:
                    try:
                        images.append(_download_image(pth))
                    except Exception as e:
                        print(f"[fallback download] {pth}: {e!r}")

        if not images:
            return {"status": "FAILED", "error": "Space returned no downloadable images", "version": VERSION}

        images_b64 = [_img_to_b64(im) for im in images]
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