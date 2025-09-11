# handler.py — RunPod Serverless SDXL → Minecraft Skin (64×64, con mapper UV)
import os, io, base64, time, traceback, fcntl, shutil
import torch
import runpod
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
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
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION")
DEFAULT_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
VAE_ID         = os.getenv("VAE_ID", DEFAULT_VAE_ID)
HF_TOKEN       = os.getenv("HF_TOKEN")

HEIGHT   = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE = float(os.getenv("GUIDANCE", "6.5"))
STEPS    = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(f"[BOOT] torch={torch.__version__} cuda={torch.cuda.is_available()} dev={DEVICE}", flush=True)

PIPE = None
_LOCK_FPATH = "/runpod-volume/sdxl_init.lock"
os.makedirs(os.path.dirname(_LOCK_FPATH), exist_ok=True)

def _force_load_vae():
    if AutoencoderKL is None:
        return None
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
        vae = None
        try:
            vae = _force_load_vae()
        except Exception as e:
            print(f"[WARN] VAE load failed: {e}", flush=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=DTYPE,
            use_safetensors=True,
            add_watermarker=None,
            token=HF_TOKEN,
            vae=vae
        )
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        except Exception as e:
            print(f"[WARN] scheduler reuse failed: {e}", flush=True)
        for attr in ("watermark", "watermarker", "safety_checker"):
            if hasattr(pipe, attr):
                try: setattr(pipe, attr, None)
                except Exception: pass
        pipe.to(DEVICE)
        PIPE = pipe
        fcntl.flock(lockf, fcntl.LOCK_UN)
        return PIPE

# --------- Mapper: highres → skin UV layout ---------
def _map_to_skin_layout(highres_img: Image.Image) -> bytes:
    """
    Recorta partes de la imagen generada y las pega en una plantilla 64x64
    siguiendo el layout clásico de Minecraft (pre-1.8/1.8).
    """
    hi = highres_img
    if hi.mode != "RGBA":
        hi = hi.convert("RGBA")
    # Downscale a 128×128 para que los bloques sean más fáciles de cortar
    base = hi.resize((128, 128), resample=Image.NEAREST)

    skin = Image.new("RGBA", (64, 64), (0,0,0,0))

    # Ejemplo simple: mapeo de cabeza (8×8 caras)
    head = base.crop((32, 0, 32+32, 32))  # bloque 32×32 en el centro arriba
    head = head.resize((8,8), Image.NEAREST)
    skin.paste(head, (8, 0))   # cara frontal
    skin.paste(head, (0, 8))   # izquierda
    skin.paste(head, (8, 8))   # detrás
    skin.paste(head, (16, 8))  # derecha
    skin.paste(head, (8,16))   # abajo
    skin.paste(head, (8,24))   # arriba

    # Torso aproximado
    torso = base.crop((48, 32, 48+32, 32+48))
    torso = torso.resize((8,12), Image.NEAREST)
    skin.paste(torso, (20, 20))

    # Piernas aproximadas
    leg = base.crop((0, 32, 0+32, 32+48))
    leg = leg.resize((4,12), Image.NEAREST)
    skin.paste(leg, (4,20))
    skin.paste(leg, (12,20))

    # Brazos aproximados
    arm = base.crop((96, 32, 96+32, 32+48))
    arm = arm.resize((4,12), Image.NEAREST)
    skin.paste(arm, (44,20))
    skin.paste(arm, (52,20))

    buf = io.BytesIO()
    skin.save(buf, format="PNG")
    return buf.getvalue()

# --------- RunPod handler ----------
def handler(event):
    try:
        inp = event.get("input", {}) or {}
        prompt   = inp.get("prompt", "minecraft skin")
        neg      = inp.get("negative_prompt") or None
        steps    = int(inp.get("steps", STEPS))
        guidance = float(inp.get("guidance", GUIDANCE))
        width    = int(inp.get("width",  WIDTH))
        height   = int(inp.get("height", HEIGHT))
        seed     = inp.get("seed")

        print(f"[RUN] prompt='{prompt[:80]}'", flush=True)

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
        ).images[0]

        png_bytes = _map_to_skin_layout(out)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"ok": True, "w": 64, "h": 64, "image_b64": b64}
    except Exception as e:
        print("[ERROR] Exception:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}

runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin"}}))
