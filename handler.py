# handler.py — RunPod Serverless SDXL → Minecraft Skin (64×64) con layout forcing + matte
import os, io, base64, time, traceback, fcntl, shutil
import torch
import runpod
from PIL import Image
from collections import Counter
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

# --------- ENV / Cache ----------
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_HOME)

# Apaga hf_transfer si no está instalado
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# --------- Modelo ----------
MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")  # opcional
DEFAULT_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
VAE_ID         = os.getenv("VAE_ID", DEFAULT_VAE_ID)
HF_TOKEN       = os.getenv("HF_TOKEN")

# --------- Parámetros por defecto (forzamos 1024) ----------
HEIGHT   = int(os.getenv("GEN_HEIGHT", "1024"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "1024"))
GUIDANCE = float(os.getenv("GUIDANCE", "5.0"))
STEPS    = int(os.getenv("STEPS",     "28"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(f"[BOOT] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={DEVICE} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
      flush=True)

PIPE = None
_LOCK_FPATH = "/runpod-volume/sdxl_init.lock"
os.makedirs(os.path.dirname(_LOCK_FPATH), exist_ok=True)

# --------- Prompt forcing ----------
PROMPT_SUFFIX = (
    " minecraft skin texture sheet, 64x64 game texture UV layout, "
    "front back left right views arranged, head top bottom left right front back, "
    "body torso arms legs mapped to template, pixel art, clean flat shading, high-contrast, "
    "no background, transparent background, template aligned, centered, full character, "
    "video game asset, texture atlas"
)

NEGATIVE_BASE = (
    "photo, realistic, 3d render, background, scenery, text, watermark, signature, "
    "portrait only, single face close-up, disorganized layout, collage, canvas, border, "
    "blurry, noisy, low-res, artifacts, misaligned, cut-off limbs, wrong aspect"
)

# --------- Utils ----------
def _disk_log():
    try:
        total, used, free = shutil.disk_usage("/")
        t2, u2, f2 = shutil.disk_usage("/runpod-volume")
        print(f"[DISK] / total={total/1e9:.1f}GB free={free/1e9:.1f}GB | "
              f"/runpod-volume total={t2/1e9:.1f}GB free={f2/1e9:.1f}GB", flush=True)
    except Exception:
        pass

def _force_load_vae():
    if AutoencoderKL is None:
        return None
    print(f"[LOAD] Loading VAE: {VAE_ID}", flush=True)
    return AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN)

def _load_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE

    with open(_LOCK_FPATH, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if PIPE is not None:
            fcntl.flock(lockf, fcntl.LOCK_UN)
            return PIPE

        _disk_log()
        t0 = time.time()
        print(f"[LOAD] Loading model: {MODEL_ID} (rev={MODEL_REVISION})", flush=True)

        vae = None
        try:
            vae = _force_load_vae()
        except Exception as e:
            print(f"[WARN] VAE load failed ({VAE_ID}): {e}", flush=True)

        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                torch_dtype=DTYPE,
                use_safetensors=True,
                add_watermarker=None,
                token=HF_TOKEN,
                vae=vae
            )
        except Exception:
            print("[ERROR] SDXL from_pretrained failed:\n" + traceback.format_exc(), flush=True)
            raise

        if getattr(pipe, "vae", None) is None:
            print("[FIX] Pipeline without VAE, forcing attach...", flush=True)
            pipe.vae = _force_load_vae()

        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        except Exception as e:
            print(f"[WARN] Scheduler reuse failed: {e}. Using default.", flush=True)

        for attr in ("watermark", "watermarker", "safety_checker"):
            if hasattr(pipe, attr):
                try:
                    setattr(pipe, attr, None)
                except Exception:
                    pass

        pipe.to(DEVICE)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] xformers not enabled: {e}", flush=True)
        pipe.enable_vae_slicing()

        PIPE = pipe
        print(f"[LOAD] Model ready in {time.time()-t0:.1f}s", flush=True)
        fcntl.flock(lockf, fcntl.LOCK_UN)
        return PIPE

def _matte_background_to_alpha(img_rgba: Image.Image, tol: int = 18) -> Image.Image:
    """
    Elimina el fondo casi uniforme (gris/rosa) convirtiéndolo a alfa=0.
    Toma el color más frecuente del borde y quita píxeles similares (tolerancia Euclídea).
    """
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")

    w, h = img_rgba.size
    pix = img_rgba.load()

    border_colors = []
    for x in range(w):
        border_colors.append(pix[x, 0][:3])
        border_colors.append(pix[x, h-1][:3])
    for y in range(h):
        border_colors.append(pix[0, y][:3])
        border_colors.append(pix[w-1, y][:3])

    (bg_r, bg_g, bg_b), _ = Counter(border_colors).most_common(1)[0]

    def close(c):
        dr = c[0] - bg_r
        dg = c[1] - bg_g
        db = c[2] - bg_b
        return (dr*dr + dg*dg + db*db) ** 0.5 <= tol

    out = Image.new("RGBA", (w, h))
    out_pix = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = pix[x, y]
            if close((r, g, b)):
                out_pix[x, y] = (r, g, b, 0)
            else:
                out_pix[x, y] = (r, g, b, a)
    return out

def _downscale_to_skin64(img: Image.Image) -> Image.Image:
    """Baja a 64×64 con vecino más cercano (mantiene píxel nítido)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Asegura cuadrado (por si el modelo devuelve 1024 no-crop)
    s = min(img.width, img.height)
    if img.width != img.height:
        img = img.crop((0, 0, s, s))
    return img.resize((64, 64), resample=Image.NEAREST)

def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ------------- RunPod handler -------------
def handler(event):
    """
    input:
      prompt (str)
      negative_prompt (str, opc; se une a NEGATIVE_BASE)
      seed (int, opc)
      steps (int, opc)
      guidance (float, opc)
      width/height (opc; default 1024)
      return_64x64 (bool, opc; default True)
      matte_bg (bool, opc; default True) — eliminar fondo a transparente
    """
    try:
        inp = event.get("input", {}) or {}
        user_prompt = inp.get("prompt", "minecraft skin")
        neg_extra   = (inp.get("negative_prompt") or "").strip()
        steps       = int(inp.get("steps", STEPS))
        guidance    = float(inp.get("guidance", GUIDANCE))
        width       = int(inp.get("width",  WIDTH))
        height      = int(inp.get("height", HEIGHT))
        r64         = bool(inp.get("return_64x64", True))
        matte_bg    = bool(inp.get("matte_bg", True))
        seed        = inp.get("seed")

        # Forzar prompts que describen el layout de skin
        prompt_full = f"{user_prompt.strip()}, {PROMPT_SUFFIX}"
        negative_full = NEGATIVE_BASE if not neg_extra else (NEGATIVE_BASE + ", " + neg_extra)

        print(f"[RUN] steps={steps} guidance={guidance} size={width}x{height} seed={seed}", flush=True)
        print(f"[PROMPT] {prompt_full[:180]}", flush=True)
        pipe = _load_pipe()

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        out = pipe(
            prompt=prompt_full,
            negative_prompt=negative_full,
            width=width, height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen
        )
        img = out.images[0]

        if matte_bg:
            img = _matte_background_to_alpha(img, tol=18)

        if r64:
            img = _downscale_to_skin64(img)

        png_bytes = _img_to_png_bytes(img)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"ok": True, "w": img.width, "h": img.height, "image_b64": b64}

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            msg = "CUDA OOM: baja GEN_WIDTH/GEN_HEIGHT a 640 o 512, steps a 22–25, y pon Concurrency=1"
            print(f"[OOM] {msg}", flush=True)
            return {"ok": False, "error": msg}
        print("[ERROR] RuntimeError:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        print("[ERROR] Exception:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}

# Arranque del worker
runpod.serverless.start({"handler": handler})

# Test local opcional
if __name__ == "__main__":
    print(handler({"input": {
        "prompt": "donald trump in a navy suit with red tie, stylized",
        "seed": 42
    }}))
