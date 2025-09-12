# handler.py — RunPod Serverless SDXL → Minecraft Skin (64×64)
import os, io, base64, time, traceback, fcntl, shutil
import runpod
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Mapper local
from mapper import to_skin_layout

try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

# --------- ENV sane defaults ----------
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_HOME)

if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")  # opcional
VAE_ID         = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
HF_TOKEN       = os.getenv("HF_TOKEN")  # opcional

HEIGHT   = int(os.getenv("GEN_HEIGHT", "1024"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "1024"))
GUIDANCE = float(os.getenv("GUIDANCE", "6.5"))
STEPS    = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(
    f"[BOOT] torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"dev={DEVICE} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
    flush=True
)

PIPE = None
_LOCK_FPATH = "/runpod-volume/sdxl_init.lock"
os.makedirs(os.path.dirname(_LOCK_FPATH), exist_ok=True)

def _disk_log():
    try:
        total, used, free = shutil.disk_usage("/")
        t2, u2, f2 = shutil.disk_usage("/runpod-volume")
        print(f"[DISK] / total={total/1e9:.1f}GB free={free/1e9:.1f}GB | "
              f"/runpod-volume total={t2/1e9:.1f}GB free={f2/1e9:.1f}GB", flush=True)
    except Exception:
        pass

def _force_load_vae():
    if AutoencoderKL is None or not VAE_ID:
        return None
    print(f"[LOAD] Loading VAE: {VAE_ID}", flush=True)
    return AutoencoderKL.from_pretrained(
        VAE_ID, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN
    )

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
            print(f"[WARN] VAE load failed: {e}", flush=True)

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

        if getattr(pipe, "vae", None) is None and VAE_ID:
            print("[FIX] Pipeline has no VAE, forcing attach…", flush=True)
            try:
                pipe.vae = _force_load_vae()
            except Exception as e:
                print(f"[WARN] Could not attach VAE: {e}", flush=True)

        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras=True
            )
        except Exception as e:
            print(f"[WARN] Scheduler reuse failed: {e}", flush=True)

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
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        PIPE = pipe
        print(f"[LOAD] Model ready in {time.time()-t0:.1f}s", flush=True)
        fcntl.flock(lockf, fcntl.LOCK_UN)
        return PIPE

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

        print(f"[RUN] prompt='{prompt[:120]}' steps={steps} "
              f"guidance={guidance} size={width}x{height} seed={seed}", flush=True)

        pipe = _load_pipe()

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width, height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen
        )
        img = result.images[0]

        if r64:
            png_bytes = to_skin_layout(img)
            w, h = 64, 64
        else:
            buf = io.BytesIO(); img.save(buf, format="PNG"); png_bytes = buf.getvalue()
            w, h = img.width, img.height

        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"ok": True, "w": w, "h": h, "image_b64": b64}

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            msg = "CUDA OOM: baja GEN_WIDTH/GEN_HEIGHT a 640 o 512, steps a 20–25, y pon Concurrency=1"
            print(f"[OOM] {msg}", flush=True)
            return {"ok": False, "error": msg}
        print("[ERROR] RuntimeError:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        print("[ERROR] Exception:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}

runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin", "seed": 42}}))
