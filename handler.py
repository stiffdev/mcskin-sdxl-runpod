# handler.py — RunPod Serverless (paridad con HF)
import os, io, base64, torch
import runpod
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# ----------------- Config vía entorno -----------------
MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")   # opcional: fija commit/tag de HF
VAE_ID         = os.getenv("VAE_ID")           # opcional: ej. "stabilityai/sdxl-vae-fp16-fix"

HEIGHT    = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH     = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE  = float(os.getenv("GUIDANCE", "6.5"))
STEPS     = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

# Determinismo razonable (misma seed -> mismo output visual)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ----------------- Carga pipeline global (mejor latencia) -----------------
vae = None
if VAE_ID:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE)

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,      # si no defines env, queda None
    torch_dtype=DTYPE,
    vae=vae,
    use_safetensors=True,
    add_watermarker=None,
).to(DEVICE)

# Scheduler como en Spaces (Karras)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras=True
)

# Alineación de “extras” para paridad con Spaces
if hasattr(pipe, "watermark"):   pipe.watermark = None
if hasattr(pipe, "watermarker"): pipe.watermarker = None
if hasattr(pipe, "safety_checker"):
    try:
        pipe.safety_checker = None
    except Exception:
        pass

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass
pipe.enable_vae_slicing()

# ----------------- Postproceso 768->64 con muestreo de centro -----------------
def _to_skin_64_png(hi: Image.Image) -> bytes:
    if hi.width != hi.height:
        s = min(hi.width, hi.height)
        hi = hi.crop((0, 0, s, s))
    if hi.mode != "RGBA":
        hi = hi.convert("RGBA")

    scale = hi.width // 64
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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------- RunPod handler -----------------
def handler(event):
    """
    event['input']:
      prompt (str), negative_prompt (str, opc), seed (int, opc),
      steps (int, opc), guidance (float, opc), return_64x64 (bool, opc)
    """
    inp = event.get("input", {}) or {}
    prompt = inp.get("prompt", "minecraft skin")
    negative = inp.get("negative_prompt") or None
    steps = int(inp.get("steps", STEPS))
    guidance = float(inp.get("guidance", GUIDANCE))
    seed = inp.get("seed")
    return_64 = bool(inp.get("return_64x64", True))

    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    img = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=WIDTH, height=HEIGHT,          # 768x768 recomendado
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=gen
    ).images[0]

    png_bytes = _to_skin_64_png(img) if return_64 else _img_to_png(img)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"image_b64": b64,
            "w": 64 if return_64 else img.width,
            "h": 64 if return_64 else img.height}

# Arranca el worker de RunPod Serverless
runpod.serverless.start({"handler": handler})

# Modo local opcional:
if __name__ == "__main__":
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin", "seed": 42}}))