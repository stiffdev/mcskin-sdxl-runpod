# handler.py — RunPod Serverless con logs y carga perezosa
import os, io, base64, time, traceback
import torch
import runpod
from PIL import Image

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

# -------- Config ----------
MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")            # opcional: tag/sha de HF
VAE_ID         = os.getenv("VAE_ID")                    # opcional: ej. stabilityai/sdxl-vae-fp16-fix
HF_TOKEN       = os.getenv("HF_TOKEN")                  # opcional si el repo fuese privado

HEIGHT = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH  = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE = float(os.getenv("GUIDANCE", "6.5"))
STEPS    = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(f"[BOOT] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"device={DEVICE} compute_cap={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
      flush=True)

PIPE = None

def _load_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE
    t0 = time.time()
    print(f"[LOAD] Loading model: {MODEL_ID} (rev={MODEL_REVISION}) VAE={VAE_ID}", flush=True)

    vae = None
    if VAE_ID and AutoencoderKL is not None:
        try:
            vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE, token=HF_TOKEN)
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
            token=HF_TOKEN
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras=True
        )
        if hasattr(pipe, "watermark"):   pipe.watermark = None
        if hasattr(pipe, "watermarker"): pipe.watermarker = None
        if hasattr(pipe, "safety_checker"):
            try: pipe.safety_checker = None
            except: pass

        pipe.to(DEVICE)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] xformers not enabled: {e}", flush=True)

        pipe.enable_vae_slicing()
        PIPE = pipe
        print(f"[LOAD] Model ready in {time.time()-t0:.1f}s", flush=True)
        return PIPE
    except Exception as e:
        print("[ERROR] Pipeline load failed:\n" + traceback.format_exc(), flush=True)
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

        print(f"[RUN] prompt='{prompt[:80]}' steps={steps} guidance={guidance} size={width}x{height} seed={seed}", flush=True)

        pipe = _load_pipe()

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        img = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width, height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen
        ).images[0]

        png_bytes = _to_skin_64_png(img) if r64 else _img_to_png(img)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"ok": True,
                "w": 64 if r64 else img.width,
                "h": 64 if r64 else img.height,
                "image_b64": b64}
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            msg = "CUDA OOM: baja GEN_WIDTH/GEN_HEIGHT a 640 o 512, steps a 25, y pon Concurrency=1"
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
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin", "seed": 42}}))