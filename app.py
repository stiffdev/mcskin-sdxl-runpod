import base64, io, os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from mapper import to_skin_64_png  # downscale fiel + validación

load_dotenv()

MODEL_ID  = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
DEVICE    = "cuda"
DTYPE     = torch.float16
HEIGHT    = int(os.getenv("GEN_HEIGHT", "768"))  # SDXL sweet spot
WIDTH     = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE  = float(os.getenv("GUIDANCE", "6.5"))
STEPS     = int(os.getenv("STEPS",     "30"))
NEG_PROMPT_DEFAULT = os.getenv("NEG_PROMPT", "")

app = FastAPI(title="Minecraft Skin Generator (SDXL)")

# --- Carga pipeline idéntica a HF ---
# Tip: si tu Space usa VAE específico, habilítalo aquí:
# from diffusers import AutoencoderKL
# vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae-fp16-fix", torch_dtype=DTYPE)

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    # vae=vae,  # <-- descomenta si en tu Space estaba fijado
    use_safetensors=True,
    add_watermarker=None,  # para igualar outputs si el Space no marca agua
).to(DEVICE)

# Scheduler recomendado / matching con Space
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras=True
)

# Optimizaciones VRAM/velocidad
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass
pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()  # usar solo si vas justo de VRAM (más lento)

class GenReq(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    return_64x64: bool = True  # por defecto devolvemos la skin lista para Minecraft

@app.get("/livez")
def livez():
    return {"ok": True, "model": MODEL_ID}

@app.post("/generate")
def generate(req: GenReq):
    g = torch.Generator(device=DEVICE)
    if req.seed is not None:
        g = g.manual_seed(req.seed)

    steps    = req.steps or STEPS
    guidance = req.guidance or GUIDANCE
    negp     = req.negative_prompt or NEG_PROMPT_DEFAULT

    img = pipe(
        prompt=req.prompt,
        negative_prompt=negp if negp else None,
        width=WIDTH, height=HEIGHT,            # 768×768 recomendado por model card
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=g
    ).images[0]  # SDXL genera “sheet” a alta resolución

    if req.return_64x64:
        png_bytes = to_skin_64_png(img)       # mapeo/downscale a 64×64 + alpha
    else:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"image_b64": b64, "w": 64 if req.return_64x64 else img.width, "h": 64 if req.return_64x64 else img.height}
