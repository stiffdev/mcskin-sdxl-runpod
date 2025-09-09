# handler.py — proxy RunPod -> HuggingFace Space (phenixrhyder/3D-Minecraft-Skin-Generator)
import os, io, base64, traceback, time, random
from typing import Any, Dict, List
from PIL import Image
import runpod
from gradio_client import Client

VERSION = "space-proxy v4"

HF_SPACE_ID  = os.getenv("HF_SPACE_ID", "phenixrhyder/3D-Minecraft-Skin-Generator")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # opcional: URL del Space
HF_TOKEN     = os.getenv("HF_TOKEN")      # muy recomendable

_client: Client | None = None

# Orden exacta del Space:
# predict(prompt, stable_diffusion_model, num_inference_steps, guidance_scale,
#         model_precision_type, seed, filename, model_3d, verbose)
def _boot():
    global _client
    print(f"[boot] {VERSION} — connecting to space: {HF_SPACE_ID} (url={HF_SPACE_URL or '-'})")
    _client = Client(HF_SPACE_URL or HF_SPACE_ID, hf_token=HF_TOKEN, verbose=False)
    try:
        info = _client.view_api(return_format="dict")
        names = list((info or {}).get("named_endpoints", {}).keys())
        print(f"[boot] view_api ok, named_endpoints={names}")
    except Exception as e:
        print("[boot] view_api failed:", repr(e))

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _open_any(x: Any) -> Image.Image | None:
    try:
        if isinstance(x, Image.Image):
            return x.convert("RGBA")
        if isinstance(x, (bytes, bytearray)):
            return Image.open(io.BytesIO(x)).convert("RGBA")
        if isinstance(x, str):
            # gradio_client descarga a un path local temporal
            return Image.open(x).convert("RGBA")
        if isinstance(x, (list, tuple)) and x:
            return _open_any(x[0])
        if isinstance(x, dict):
            p = x.get("path") or x.get("name")
            if isinstance(p, str):
                return Image.open(p).convert("RGBA")
    except Exception:
        return None
    return None

def _flatten_images(result: Any) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    def push(v):
        im = _open_any(v)
        if im:
            imgs.append(im)
    if isinstance(result, list):
        for v in result:
            push(v)
    else:
        push(result)
    if not imgs and isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, list):
            for v in first:
                push(v)
        else:
            push(first)
    return imgs

def _build_inputs(p: Dict[str, Any]) -> List[Any]:
    prompt = (p.get("prompt") or p.get("text") or p.get("input") or "").strip()
    if not prompt:
        raise ValueError("missing prompt")

    sd_model  = p.get("stable_diffusion_model") or "xl"
    steps     = float(p.get("steps") or p.get("num_inference_steps") or 28)
    cfg       = float(p.get("guidance_scale") or p.get("cfg") or 6.5)
    precision = p.get("model_precision_type") or "fp16"

    # Sólo pasamos seed si el cliente la manda explícitamente
    seed_raw = p.get("seed", None)
    seed_val = None
    if seed_raw not in (None, "", "random", "rnd", "auto"):
        try:
            seed_val = float(seed_raw)
        except Exception:
            seed_val = None  # ignora seeds inválidas

    # Nombre único por request para evitar caché del Space / archivo
    uniq = int(time.time() * 1000)
    filename = p.get("filename") or f"skin-{uniq}.png"

    model_3d = bool(p.get("model_3d", True))
    verbose  = bool(p.get("verbose", False))

    # Orden posicional exacta:
    # prompt, stable_diffusion_model, num_inference_steps, guidance_scale,
    # model_precision_type, seed, filename, model_3d, verbose
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

        # Evita estado pegado en la sesión de Gradio
        try:
            _client.reset_session()
        except Exception:
            pass

        t0 = time.time()
        result = _client.predict(*inputs, api_name="/predict")
        dt = time.time() - t0
        print(f"[run] space call done in {dt:.2f}s")

        imgs = _flatten_images(result)
        if not imgs:
            return {"status": "FAILED", "error": "Space returned no images"}

        images_b64 = [_to_b64(im) for im in imgs]
        return {
            "status": "COMPLETED",
            "images": images_b64,
            "output": {"images": images_b64},
            "meta": {
                "elapsed_sec": dt,
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