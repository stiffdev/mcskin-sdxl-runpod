# handler.py
import os, io, base64, traceback, time
from PIL import Image
import runpod
from gradio_client import Client, handle_file
from typing import Any, Dict, List

HF_SPACE_ID = os.getenv("HF_SPACE_ID", "phenixrhyder/3D-Minecraft-Skin-Generator")
HF_TOKEN    = os.getenv("HF_TOKEN")  # opcional, pero MUY recomendable para evitar rate limit
SPACE_URL   = os.getenv("HF_SPACE_URL")  # opcional, p.ej. https://phenixrhyder-3d-minecraft-skin-generator.hf.space/

client: Client | None = None
api_name: str | None = None

def _boot():
    global client, api_name
    print(f"[boot] connecting to space: {HF_SPACE_ID}")
    # Con token si lo tienes
    client = Client(SPACE_URL or HF_SPACE_ID, hf_token=HF_TOKEN, verbose=False)

    # Descubre endpoints disponibles
    try:
        info = client.view_api(return_format="dict")  # {'named_endpoints': {'/predict': {...}, ...}}
        named = info.get("named_endpoints") or {}
        # Heurística: elige el primero que produzca imagen o listado con imagen
        # Si no sabemos, probamos /predict
        candidates = list(named.keys()) or ["/predict"]
        api_name = None
        for k in candidates:
            api_name = k
            break
        print(f"[boot] selected api_name={api_name}")
    except Exception as e:
        print("[boot] view_api failed:", repr(e))
        api_name = "/predict"  # fallback

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def _result_to_images(result: Any) -> List[Image.Image]:
    """
    Convierte lo que devuelve el Space en lista de PIL Images.
    Gradio puede devolver:
      - PIL.Image
      - ruta de archivo (str)
      - bytes
      - lista de lo anterior
    """
    def load_one(x) -> Image.Image | None:
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(x)).convert("RGBA")
            except Exception:
                return None
        if isinstance(x, str):
            # path temp del Space
            try:
                return Image.open(x).convert("RGBA")
            except Exception:
                return None
        return None

    imgs: List[Image.Image] = []
    if isinstance(result, list):
        for v in result:
            im = load_one(v)
            if im: imgs.append(im)
    else:
        im = load_one(result)
        if im: imgs.append(im)

    return imgs

def handler(event: Dict[str, Any]):
    global client, api_name
    try:
        if client is None:
            _boot()

        p = (event.get("input") or {}) if isinstance(event, dict) else {}
        if p.get("warmup"):  # healthcheck
            return {"status": "WARM", "ok": True}

        # Payload básico esperado desde tu app/Worker
        prompt         = (p.get("prompt") or p.get("text") or p.get("input") or "").strip()
        negative       = (p.get("negative_prompt") or "").strip() or None
        seed           = p.get("seed", None)
        steps          = p.get("steps", None)
        num_images     = p.get("num_images", None)
        guidance_scale = p.get("guidance_scale", p.get("cfg", None))

        # Preparamos kwargs: solo pasamos lo que exista y no sea None.
        # Distintos Spaces cambian el nombre de parámetros; por eso probamos
        # varias claves comúnmente usadas.
        # Gradio ignora kwargs desconocidos.
        candidate_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": seed,
            "steps": steps or p.get("num_inference_steps"),
            "num_images": num_images,
            "guidance_scale": guidance_scale,
        }
        # filtra None
        gr_kwargs = {k: v for k, v in candidate_kwargs.items() if v not in (None, "", [])}

        print(f"[run] api={api_name} kwargs={gr_kwargs}")

        # Llamada a la API del Space
        t0 = time.time()
        result = client.predict(api_name=api_name, **gr_kwargs)
        dt = time.time() - t0
        print(f"[run] space call done in {dt:.2f}s")

        imgs = _result_to_images(result)
        if not imgs:
            # A veces el Space devuelve (data, extras)
            if isinstance(result, (list, tuple)) and result:
                imgs = _result_to_images(result[0])

        if not imgs:
            return {"status": "FAILED", "error": "Space returned no images"}

        images_b64 = [_to_b64(im) for im in imgs]
        return {
            "status": "COMPLETED",
            "images": images_b64,
            "output": {"images": images_b64, "prompt": prompt},
            "meta": {"elapsed_sec": dt, "count": len(images_b64)}
        }

    except Exception as e:
        print("[handler] FAILED:", repr(e))
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

runpod.serverless.start({"handler": handler})