# handler.py — RunPod proxy -> HF Space (phenixrhyder/3D-Minecraft-Skin-Generator)
import os, io, base64, time, traceback
from typing import Any, Dict, List
from PIL import Image
import requests
import runpod
from gradio_client import Client

VERSION = "space-proxy v5"

HF_SPACE_ID  = os.getenv("HF_SPACE_ID", "phenixrhyder/3D-Minecraft-Skin-Generator")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")   # opcional: URL directa del Space
HF_TOKEN     = os.getenv("HF_TOKEN")       # MUY recomendable

_client: Client | None = None

def _boot():
    global _client
    print(f"[boot] {VERSION} — connecting to space: {HF_SPACE_ID} (url={HF_SPACE_URL or '-'})")
    _client = Client(HF_SPACE_URL or HF_SPACE_ID, hf_token=HF_TOKEN, verbose=False)
    try:
        info = _client.view_api(return_format="dict")
        print(f"[boot] view_api ok, named_endpoints={list((info or {}).get('named_endpoints', {}).keys())}")
    except Exception as e:
        print("[boot] view_api failed:", repr(e))

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def _download_image(url: str) -> Image.Image:
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    # algunos Spaces requieren Referer para /file=
    if HF_SPACE_URL:
        headers["Referer"] = HF_SPACE_URL if HF_SPACE_URL.startswith("http") else f"https://{HF_SPACE_URL}"
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def _extract_file_urls(obj: Any) -> List[str]:
    """Busca strings tipo 'https://...hf.space/file=...' en estructuras anidadas."""
    hits: List[str] = []
    def walk(x: Any):
        if isinstance(x, str):
            if "/file=" in x and x.startswith("http"):
                hits.append(x)
        elif isinstance(x, dict):
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

    # Solo respetamos seed si la mandas explícita (para no “anclar” resultados)
    seed_raw = p.get("seed", None)
    try:
        seed_val = None if seed_raw in (None, "", "random", "rnd", "auto") else float(seed_raw)
    except Exception:
        seed_val = None

    # filename único para evitar devolver el archivo anterior por caché
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
        short_prompt = (prompt[:100] + "…") if len(prompt) > 100 else prompt
        print(f"[run] api=/predict prompt={short_prompt!r} sd={sd_model} steps={steps} cfg={cfg} "
              f"prec={precision} seed={seed} file={filename} 3d={model_3d} verbose={verbose}")

        # A veces ayuda resetear la sesión para evitar “estado pegado”
        try:
            _client.reset_session()
        except Exception:
            pass

        # 1) Ejecuta vía cola y NO dejes que el client descargue por su cuenta
        job = _client.submit(*inputs, api_name="/predict")
        result = job.outputs()  # espera y devuelve estructura JSON-like

        # 2) Extrae URLs /file= y bájalas tú (con token y referer si hace falta)
        urls = _extract_file_urls(result)
        images: List[Image.Image] = []
        for u in urls:
            try:
                images.append(_download_image(u))
            except Exception as e:
                print(f"[download] fetch failed {u}: {e!r}")

        # Fallback: si no encontramos URLs, intenta el predict “normal”
        if not images:
            try:
                print("[fallback] calling client.predict()")
                pred = _client.predict(*inputs, api_name="/predict")
                # pred suele traer rutas/urls; inténtalo igual
                for u in _extract_file_urls(pred):
                    try:
                        images.append(_download_image(u))
                    except Exception as e:
                        print(f"[fallback download] {u}: {e!r}")
            except Exception as e:
                print("[fallback] predict failed:", repr(e))

        if not images:
            return {"status": "FAILED", "error": "Space returned no downloadable images", "version": VERSION}

        images_b64 = [_to_b64(im) for im in images]
        return {
            "status": "COMPLETED",
            "images": images_b64,
            "output": {"images": images_b64},
            "meta": {
                "elapsed_sec": getattr(job, "time", None),
                "count": len(images_b64),
                "sd_model": sd_model,
                "steps": steps,
                "cfg": cfg,
                "precision": precision,
                "seed": seed,
                "filename": filename,
            },
        }

    except Exception as e:
        print("[handler] FAILED:", repr(e))
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e), "version": VERSION}

runpod.serverless.start({"handler": handler})