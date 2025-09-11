# handler.py — RunPod Serverless SDXL → Minecraft Skin (64×64) estable
import os, io, base64, time, traceback, fcntl, shutil, collections
import torch
import runpod
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

# ----------------- ENV sane defaults + cache persistente -----------------
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_HOME)

# Si HF transfer no está instalado, no lo uses (evita fallo de import)
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ----------------- Configurable por ENV -----------------
MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")  # opcional
DEFAULT_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"    # estable para SDXL fp16
VAE_ID         = os.getenv("VAE_ID", DEFAULT_VAE_ID)
HF_TOKEN       = os.getenv("HF_TOKEN")              # si fuese privado

# MUY IMPORTANTE: SDXL de ese repo se comporta mejor a 1024
HEIGHT   = int(os.getenv("GEN_HEIGHT", "1024"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "1024"))

# Los valores de CFG/steps que mejor mantienen el layout
GUIDANCE = float(os.getenv("GUIDANCE", "3.0"))   # CFG bajito ayuda al atlas
STEPS    = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _gpu_name():
    try: return torch.cuda.get_device_name(0)
    except: return "cpu"

print(f"[BOOT] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={DEVICE} name={_gpu_name()}", flush=True)

PIPE = None
_LOCK_FPATH = "/runpod-volume/sdxl_init.lock"
os.makedirs(os.path.dirname(_LOCK_FPATH), exist_ok=True)

# ----------------- Utilidades -----------------
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
    return AutoencoderKL.from_pretrained(
        VAE_ID, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN
    )

def _load_pipe():
    """
    Carga única con lock. VAE forzado para evitar `NoneType.config`.
    """
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

        # Si por lo que sea no quedó VAE:
        if getattr(pipe, "vae", None) is None:
            print("[FIX] Pipeline arrived without VAE, forcing attach...", flush=True)
            pipe.vae = _force_load_vae()

        # Scheduler Karras para SDXL
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras=True
            )
        except Exception as e:
            print(f"[WARN] Scheduler reuse failed: {e}. Using default.", flush=True)

        # Desactivar “extras”
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

# ----------------- Mapper / Post-proc -----------------
def _most_common_color(img: Image.Image) -> tuple:
    """
    Encuentra el color de fondo dominante (ignorando zonas muy pequeñas).
    Trabajamos sobre una imagen reducida para ser rápidos.
    """
    small = img.resize((64, 64), Image.NEAREST).convert("RGBA")
    cnt = collections.Counter(small.getdata())
    # ignorar transparencia si ya la hubiera
    cnt.pop((0,0,0,0), None)
    return cnt.most_common(1)[0][0] if cnt else (0, 0, 0, 255)

def _make_bg_transparent(img: Image.Image) -> Image.Image:
    """
    Heurística robusta: toma el color dominante de fondo y lo hace transparente
    con tolerancia (por si el fondo varía un poco).
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = _most_common_color(img)
    px = img.load()
    w, h = img.size
    tol = 18  # tolerancia en cada canal
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if (abs(r - bg[0]) <= tol and
                abs(g - bg[1]) <= tol and
                abs(b - bg[2]) <= tol):
                px[x, y] = (r, g, b, 0)  # transparente
    return img

def _to_skin_64_png(hi: Image.Image) -> bytes:
    """
    Reescalado a 64×64 con NEAREST + limpieza de fondo para transparencia.
    """
    # forzar cuadrado por si el modelo devuelve 1024×768 o similar
    if hi.width != hi.height:
        s = min(hi.width, hi.height)
        hi = hi.crop((0, 0, s, s))
    hi = _make_bg_transparent(hi)

    lo = hi.resize((64, 64), resample=Image.NEAREST)
    if lo.mode != "RGBA":
        lo = lo.convert("RGBA")
    buf = io.BytesIO()
    lo.save(buf, format="PNG")
    return buf.getvalue()

def _img_to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------- Prompt helper -----------------
_BASE_POSITIVE = (
    "minecraft skin texture atlas, character skin UV layout, head body arms legs, "
    "front back left right faces arranged correctly, pixel art style, clean edges, "
    "no background, transparent background, plain template, 64x64 compatible"
)
_BASE_NEGATIVE = (
    "wrong layout, misaligned atlas, cropped, photo, 3d render, perspective, watermark, "
    "text, logo, ui, extra limbs, background, shadows, noise, blurry, jpeg artifacts"
)

def _compose_prompts(user_prompt: str):
    # SDXL acepta prompt principal y prompt_2 (de estilo). Damos el base en ambos.
    p = f"{user_prompt}, {_BASE_POSITIVE}"
    p2 = "flat pixel art, simple color palette, clean lines, high contrast"
    n = _BASE_NEGATIVE
    return p, p2, n

# ----------------- RunPod handler -----------------
def handler(event):
    """
    input:
      prompt (str)
      negative_prompt (str, opc)
      seed (int, opc)
      steps (int, opc)
      guidance (float, opc)
      width/height (opc; por defecto ENV - usar 1024x1024)
      return_64x64 (bool, opc; default True)
    """
    try:
        inp = event.get("input", {}) or {}
        user_prompt = inp.get("prompt", "minecraft skin")
        neg_user    = inp.get("negative_prompt") or None
        steps       = int(inp.get("steps", STEPS))
        guidance    = float(inp.get("guidance", GUIDANCE))
        width       = int(inp.get("width",  WIDTH))
        height      = int(inp.get("height", HEIGHT))
        r64         = bool(inp.get("return_64x64", True))
        seed        = inp.get("seed")

        # Recomendación dura: 1024x1024
        if width != height:
            m = min(width, height)
            width = height = m
        if width < 1024:
            width = height = 1024

        pp, pp2, nn = _compose_prompts(user_prompt)
        if neg_user:
            nn = f"{nn}, {neg_user}"

        print(f"[RUN] prompt='{user_prompt[:120]}' steps={steps} "
              f"guidance={guidance} size={width}x{height} seed={seed}", flush=True)

        pipe = _load_pipe()

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        out = pipe(
            prompt=pp,
            prompt_2=pp2,                # SDXL dual prompt para estilo
            negative_prompt=nn,
            width=width, height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen
        )
        img = out.images[0]

        png_bytes = _to_skin_64_png(img) if r64 else _img_to_png(img)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"ok": True,
                "w": 64 if r64 else img.width,
                "h": 64 if r64 else img.height,
                "image_b64": b64}
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            msg = "CUDA OOM: baja GEN_WIDTH/GEN_HEIGHT a 1024 (ya está), steps a 20–25, y pon Concurrency=1"
            print(f"[OOM] {msg}", flush=True)
            return {"ok": False, "error": msg}
        print("[ERROR] RuntimeError:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        print("[ERROR] Exception:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}

# Arranque del worker
runpod.serverless.start({"handler": handler})

# Prueba local opcional
if __name__ == "__main__":
    print(handler({"input": {"prompt": "donald trump suit, usa flag motif", "seed": 42}}))
