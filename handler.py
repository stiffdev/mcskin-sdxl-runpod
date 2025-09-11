# handler.py — RunPod serverless, genera skin y aplica mapper
import os, io, base64, traceback, torch, runpod
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from mapper import to_skin_layout

MODEL_ID = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
DEVICE   = "cuda"
DTYPE    = torch.float16

HEIGHT   = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE = float(os.getenv("GUIDANCE", "6.5"))
STEPS    = int(os.getenv("STEPS",     "30"))

pipe = None
def load_pipe():
    global pipe
    if pipe is None:
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
        except: pass
        pipe.enable_vae_slicing()
    return pipe

def handler(event):
    try:
        inp = event.get("input", {}) or {}
        prompt   = inp.get("prompt", "minecraft skin")
        neg      = inp.get("negative_prompt")
        steps    = int(inp.get("steps", STEPS))
        guidance = float(inp.get("guidance", GUIDANCE))
        width    = int(inp.get("width", WIDTH))
        height   = int(inp.get("height", HEIGHT))
        seed     = inp.get("seed")

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        pipe = load_pipe()
        img = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width, height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen
        ).images[0]

        # aplicar mapper
        skin = to_skin_layout(img)

        buf = io.BytesIO()
        skin.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"ok": True, "w": 64, "h": 64, "image_b64": b64}
    except Exception as e:
        return {"ok": False, "error": traceback.format_exc()}

runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin"}}))