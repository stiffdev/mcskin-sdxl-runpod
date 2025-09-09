import os, io, base64, random, traceback
from typing import List, Optional
from PIL import Image
import torch, runpod

from diffusers import (
    StableDiffusionXLPipeline, StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Puedes apuntar a rutas locales (si algún día horneas modelos) o al Hub
BASE_MODEL_PATH_SDXL = os.getenv("BASE_MODEL_PATH_SDXL", "/models/sdxl")
BASE_MODEL_PATH_SD2  = os.getenv("BASE_MODEL_PATH_SD2",  "/models/sd2")
MODEL_ID_SDXL        = os.getenv("MODEL_ID_SDXL", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_ID_SD2         = os.getenv("MODEL_ID_SD2",  "monadical-labs/minecraft-skin-generator")
HF_TOKEN             = os.getenv("HF_TOKEN", None)

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def quantize(img: Image.Image, colors: int) -> Image.Image:
    if not colors or colors <= 0: return img
    return img.convert("RGB").convert(
        "P", palette=Image.ADAPTIVE, colors=int(colors), dither=Image.Dither.NONE
    ).convert("RGBA")

def downscale_to_skin(img: Image.Image) -> Image.Image:
    return img.convert("RGBA").resize((64, 64), resample=Image.NEAREST)

def sanitize_txt(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = s.strip()
    return s if s else None

pipe_sdxl = None
pipe_sd2  = None

def _set_scheduler(pipe):
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        print("[boot] scheduler: Euler A")
    except Exception as e:
        print("[boot] scheduler keep default:", repr(e))
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe.to(DEVICE)

def load_sdxl():
    """Carga SDXL desde carpeta local si existe; si no, desde Hugging Face Hub."""
    global pipe_sdxl
    if pipe_sdxl is not None: return pipe_sdxl

    if os.path.isdir(BASE_MODEL_PATH_SDXL):
        print("[boot] loading SDXL (local):", BASE_MODEL_PATH_SDXL)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL_PATH_SDXL, torch_dtype=DTYPE, local_files_only=True
        )
    else:
        print("[boot] loading SDXL (hub):", MODEL_ID_SDXL)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID_SDXL, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN
        )
    pipe_sdxl = _set_scheduler(pipe)
    return pipe_sdxl

def load_sd2():
    """Carga SD2 desde carpeta local si existe; si no, desde Hugging Face Hub."""
    global pipe_sd2
    if pipe_sd2 is not None: return pipe_sd2

    if os.path.isdir(BASE_MODEL_PATH_SD2):
        print("[boot] loading SD2 (local):", BASE_MODEL_PATH_SD2)
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH_SD2, torch_dtype=DTYPE, safety_checker=None, local_files_only=True
        )
    else:
        print("[boot] loading SD2 (hub):", MODEL_ID_SD2)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID_SD2, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN, safety_checker=None
        )
    pipe_sd2 = _set_scheduler(pipe)
    return pipe_sd2

ATLAS_SUFFIX = ", minecraft texture atlas 64x64, pixel art, flat lighting, transparent background, no logo, no watermark"
NEGATIVE_DEFAULT = "photo, background, text, logo, watermark, gradient, blur, noisy, low quality, artifacts"

def generate(
    pipe, prompt: str,
    steps: int = 28, cfg: float = 6.5,
    negative_prompt: Optional[str] = None,
    num_images: int = 1,
    seed: Optional[int] = None,
    quantize_colors: int = 28
) -> List[Image.Image]:
    prompt = prompt.strip() + ATLAS_SUFFIX
    negative_prompt = sanitize_txt(negative_prompt) or NEGATIVE_DEFAULT
    if seed in (None, "", "random", "auto"):
        seed = random.randint(1, 2**31-1)
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    out = pipe(
        prompt=[prompt]*int(num_images),
        negative_prompt=[negative_prompt]*int(num_images),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=g
    ).images
    imgs = []
    for im in out:
        im = downscale_to_skin(im)
        if quantize_colors: im = quantize(im, quantize_colors)
        imgs.append(im)
    return imgs

def handler(event):
    try:
        p = (event.get("input") or {})
        if p.get("warmup"):
            return {"ok": True, "status": "WARM"}

        model = (p.get("model") or "sdxl").strip().lower()
        pipe = load_sd2() if model == "sd2" else load_sdxl()

        prompt  = sanitize_txt(p.get("prompt")) or "minecraft character, pixel art"
        steps   = max(16, min(int(p.get("steps", 28)), 40))
        cfg     = max(3.0, min(float(p.get("cfg", 6.5)), 12.0))
        n       = max(1, min(int(p.get("num_images", 1)), 4))
        seedraw = p.get("seed", None)
        seed    = None if seedraw in (None, "", "random", "auto") else int(seedraw)
        neg     = p.get("negative_prompt", None)
        qcols   = int(p.get("quantize_colors", 28))

        print(f"[run] model={model} steps={steps} cfg={cfg} n={n} seed={seed} prompt={prompt!r}")

        imgs = generate(pipe, prompt, steps=steps, cfg=cfg,
                        negative_prompt=neg, num_images=n,
                        seed=seed, quantize_colors=qcols)

        images_b64 = [to_b64(im) for im in imgs]
        meta = {"model": model, "steps": steps, "cfg": cfg, "seed": seed, "count": len(images_b64)}

        return {
            "status": "COMPLETED",
            "prompt": prompt,
            "images": images_b64,
            "meta": meta,
            "output": {"images": images_b64, "meta": meta}
        }
    except Exception as e:
        print("[handler] FAILED:", repr(e))
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

runpod.serverless.start({"handler": handler})