# handler.py — RunPod Serverless
import os, io, base64, torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# --- Config (mismas que usabas en FastAPI) ---
MODEL_ID  = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
HEIGHT    = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH     = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE  = float(os.getenv("GUIDANCE", "6.5"))
STEPS     = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

# --- Preload global (mejor latencia) ---
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    use_safetensors=True,
    add_watermarker=None
).to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras=True
)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass
pipe.enable_vae_slicing()

def _to_skin_64_png(img: Image.Image) -> bytes:
    if img.width != img.height:
        s = min(img.width, img.height)
        img = img.crop((0, 0, s, s))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img64 = img.resize((64, 64), resample=Image.NEAREST)
    buf = io.BytesIO()
    img64.save(buf, format="PNG")
    return buf.getvalue()

# --- RunPod handler ---
def handler(event):
    """
    event['input'] debe contener:
      prompt (str), optional: negative_prompt (str), seed (int), steps (int),
      guidance (float), return_64x64 (bool)
    """
    inp = event.get("input", {}) or {}
    prompt = inp.get("prompt", "minecraft skin")
    neg    = inp.get("negative_prompt") or None
    steps  = int(inp.get("steps", STEPS))
    guide  = float(inp.get("guidance", GUIDANCE))
    r64    = bool(inp.get("return_64x64", True))
    seed   = inp.get("seed")

    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=neg,
        width=WIDTH, height=HEIGHT,
        guidance_scale=guide,
        num_inference_steps=steps,
        generator=gen
    ).images[0]

    png_bytes = _to_skin_64_png(out) if r64 else _to_skin_64_png(out)  # normalmente queremos 64x64
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"image_b64": b64, "w": 64, "h": 64}

# Entrypoint serverless
if __name__ == "__main__":
    # Modo local opcional de prueba:
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin", "seed": 42}}))
