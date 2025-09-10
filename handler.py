# handler.py — RunPod Serverless: caché en volumen + scheduler defensivo (sin .config)
import os, io, base64, time, traceback, tempfile, shutil
import torch
import runpod
from PIL import Image

# ---------- RUTAS DE CACHÉ Y TMP EN EL VOLUMEN ----------
VOLUME_ROOT = os.getenv("VOLUME_ROOT", "/runpod-volume")
HF_CACHE_DIR = os.path.join(VOLUME_ROOT, "hf-cache")
TMP_DIR      = os.path.join(VOLUME_ROOT, "tmp")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("DIFFUSERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TORCH_HOME", HF_CACHE_DIR)

os.environ.setdefault("TMPDIR", TMP_DIR)
os.environ.setdefault("TEMP", TMP_DIR)
os.environ.setdefault("TMP", TMP_DIR)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Evita error de hf_transfer si no está instalado
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ---------- IMPORTS QUE USAN LAS VARS DE ENTORNO ----------
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

# -------- Config ----------
MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")  # opcional
VAE_ID         = os.getenv("VAE_ID")          # opcional, ej: stabilityai/sdxl-vae-fp16-fix
HF_TOKEN       = os.getenv("HF_TOKEN")        # opcional si repo privado

HEIGHT   = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE = float(os.getenv("GUIDANCE", "6.5"))
STEPS    = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _print_space():
    def _human(n):
        for u in ["B","KB","MB","GB","TB"]:
            if n < 1024: return f"{n:.1f}{u}"
            n/=1024
        return f"{n:.1f}PB"
    try:
        def _df(path):
            st = os.statvfs(path)
            total = st.f_frsize * st.f_blocks
            free  = st.f_frsize * st.f_bavail
            return _human(total), _human(free)
        total_root, free_root = _df("/")
        total_vol,  free_vol  = _df(VOLUME_ROOT)
        print(f"[DISK] / total={total_root} free={free_root} | {VOLUME_ROOT} total={total_vol} free={free_vol}", flush=True)
    except Exception as e:
        print(f"[DISK] error leyendo espacio: {e}", flush=True)

print(f"[BOOT] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={DEVICE} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
_print_space()
print(f"[PATHS] HF_CACHE_DIR={HF_CACHE_DIR} TMP_DIR={TMP_DIR}", flush=True)

PIPE = None

def _safe_set_scheduler(pipe):
    """
    No uses pipe.scheduler.config. Si no hay scheduler, crea DPMSolverMultistepScheduler()
    con valores por defecto (sin from_config).
    """
    try:
        if getattr(pipe, "scheduler", None) is None:
            print("[SCHED] pipeline.sheduler is None -> creating DPMSolverMultistepScheduler()", flush=True)
            pipe.scheduler = DPMSolverMultistepScheduler()  # defaults internos
        else:
            # Si existe, mejor no tocarlo para evitar .config en repos sin scheduler válido.
            print(f"[SCHED] using existing scheduler class={pipe.scheduler.__class__.__name__}", flush=True)
    except Exception as e:
        print(f"[SCHED][WARN] failed to ensure scheduler: {e}. Falling back to new DPMSolverMultistepScheduler().", flush=True)
        try:
            pipe.scheduler = DPMSolverMultistepScheduler()
        except Exception as ee:
            print(f"[SCHED][ERROR] fallback scheduler creation failed: {ee}", flush=True)
            raise

def _load_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE
    t0 = time.time()
    print(f"[LOAD] Loading model: {MODEL_ID} (rev={MODEL_REVISION}) VAE={VAE_ID}", flush=True)

    vae = None
    if VAE_ID and AutoencoderKL is not None:
        try:
            vae = AutoencoderKL.from_pretrained(
                VAE_ID, torch_dtype=DTYPE, cache_dir=HF_CACHE_DIR, token=HF_TOKEN
            )
            print("[LOAD] VAE loaded", flush=True)
        except Exception as e:
            print(f"[WARN] VAE load failed: {e}", flush=True)

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=DTYPE,
            vae=vae,
            use_safetensors=True,
            add_watermarker=None,
            token=HF_TOKEN,
            cache_dir=HF_CACHE_DIR,
            local_files_only=False
        )

        # Scheduler defensivo (SIN tocar .config)
        _safe_set_scheduler(pipe)

        # Desactiva watermark/safety si existen
        for attr in ("watermark", "watermarker", "safety_checker"):
            if hasattr(pipe, attr):
                try: setattr(pipe, attr, None)
                except Exception: pass

        pipe.to(DEVICE)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] xformers not enabled: {e}", flush=True)

        pipe.enable_vae_slicing()

        print(f"[LOAD] Model ready in {time.time()-t0:.1f}s", flush=True)
        _print_space()
        PIPE = pipe
        return PIPE

    except Exception:
        print("[ERROR] Pipeline load failed:\n" + traceback.format_exc(), flush=True)
        try:
            _print_space()
            # Limpia temporales si quedó algo corrupto ocupando espacio
            if os.path.isdir(TMP_DIR):
                for name in os.listdir(TMP_DIR):
                    p = os.path.join(TMP_DIR, name)
                    try:
                        if os.path.isfile(p) or os.path.islink(p):
                            os.unlink(p)
                        else:
                            shutil.rmtree(p, ignore_errors=True)
                    except Exception:
                        pass
            _print_space()
        except Exception:
            pass
        raise

def _to_skin_64_png(hi: Image.Image) -> bytes:
    if hi.width != hi.height:
        s = min(hi.width, hi.height)
        hi = hi.crop((0, 0, s, s))
    if hi.mode != "RGBA":
        hi = hi.convert("RGBA")

    scale = max(1, hi.width // 64)
    if scale >= 2:
        lo = Image.new("RGBA", (64, 64))
        src = hi.load(); dst = lo.load()
        for y in range(64):
            sy = min(int((y + 0.5) * scale), hi.height - 1)
            for x in range(64):
                sx = min(int((x + 0.5) * scale), hi.width - 1)
                dst[x, y] = src[sx, sy]
    else:
        lo = hi.resize((64, 64), resample=Image.NEAREST)

    buf = io.BytesIO()
    lo.save(buf, format="PNG")
    return buf.getvalue()

def _img_to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

# ------------- RunPod handler -------------
def handler(event):
    """
    input:
      prompt (str)
      negative_prompt (str, opc)
      seed (int, opc)
      steps (int, opc)
      guidance (float, opc)
      width/height (opc; por defecto ENV)
      return_64x64 (bool, opc; default True)
    """
    tempfile.tempdir = TMP_DIR

    try:
        inp = event.get("input", {}) or {}
        prompt   = inp.get("prompt", "minecraft skin")
        neg      = inp.get("negative_prompt") or None
        steps    = int(inp.get("steps", STEPS))
        guidance = float(inp.get("guidance", GUIDANCE))
        width    = int(inp.get("width",  WIDTH))
        height   = int(inp.get("height", HEIGHT))
        r64      = bool(inp.get("return_64x64", True))
        seed     = inp.get("seed")

        print(f"[RUN] prompt='{prompt[:120]}' steps={steps} guidance={guidance} size={width}x{height} seed={seed}", flush=True)
        _print_space()

        pipe = _load_pipe()

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        out = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width, height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen
        )
        img = out.images[0]

        png_bytes = _to_skin_64_png(img) if r64 else _img_to_png(img)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        print("[RUN] done", flush=True)
        return {
            "ok": True,
            "w": 64 if r64 else img.width,
            "h": 64 if r64 else img.height,
            "image_b64": b64
        }

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            msg = "CUDA OOM: baja GEN_WIDTH/GEN_HEIGHT a 640 o 512, steps a 25, y pon Concurrency=1"
            print(f"[OOM] {msg}", flush=True)
            return {"ok": False, "error": msg}
        print("[ERROR] RuntimeError:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}
    except OSError as e:
        if "No space left on device" in str(e):
            print("[DISK] No space left. Revisa volumen/cachés. Estado actual:", flush=True)
            _print_space()
        print("[ERROR] OSError:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        print("[ERROR] Exception:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}

# Arranque del worker
runpod.serverless.start({"handler": handler})

# Test local opcional
if __name__ == "__main__":
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin", "seed": 42}}))
