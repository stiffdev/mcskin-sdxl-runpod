# handler.py — RunPod Serverless: robusto, sin .config, con fallback a SDXL base y LoRA opcional
import os, io, base64, time, traceback
import torch
import runpod
from PIL import Image

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

# ----------------------------
# Parcheo hf_transfer (opcional)
# ----------------------------
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
    try:
        import hf_transfer  # noqa: F401
        print("[BOOT] hf_transfer presente (fast download ON)", flush=True)
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("[BOOT] HF_HUB_ENABLE_HF_TRANSFER=1 pero 'hf_transfer' no está instalado -> lo desactivo", flush=True)

HF_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE")
if HF_CACHE:
    try:
        os.makedirs(HF_CACHE, exist_ok=True)
        print(f"[BOOT] HUGGINGFACE_HUB_CACHE={HF_CACHE}", flush=True)
    except Exception as _e:
        print(f"[WARN] No pude crear cache dir {HF_CACHE}: {_e}", flush=True)

torch.set_grad_enabled(False)

# ----------------
# ENV y constantes
# ----------------
MODEL_ID         = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION   = os.getenv("MODEL_REVISION")
BASE_SDXL_ID     = os.getenv("BASE_SDXL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN         = os.getenv("HF_TOKEN")
VAE_ID           = os.getenv("VAE_ID")  # opcional
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")  # habitual en LoRA
TRY_LORA         = os.getenv("TRY_LORA", "1") == "1"  # intenta aplicar LoRA del repo MODEL_ID

HEIGHT   = int(os.getenv("GEN_HEIGHT", "768"))
WIDTH    = int(os.getenv("GEN_WIDTH",  "768"))
GUIDANCE = float(os.getenv("GUIDANCE", "6.5"))
STEPS    = int(os.getenv("STEPS",     "30"))

DTYPE  = torch.float16
DEVICE = "cuda"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(
    f"[BOOT] torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
    flush=True
)

PIPE = None


# --------------------
# Scheduler “blindado”
# --------------------
def _attach_scheduler(pipe):
    """
    Nunca usamos .config de ningún sitio (evita 'NoneType.config').
    Creamos un DPMSolverMultistepScheduler con parámetros típicos SDXL.
    """
    pipe.scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++",
        use_karras=True,
        steps_offset=1
    )
    print("[SCHED] DPMSolverMultistepScheduler (fallback SDXL) adjuntado", flush=True)


# -----------------------
# Carga robusta del modelo
# -----------------------
def _load_pipe():
    """
    Estrategia:
    1) Intentar cargar MODEL_ID como pipeline SDXL completa.
    2) Si falla, cargar SDXL base.
       2a) Si TRY_LORA=1, intentar aplicar LoRA desde MODEL_ID (LORA_WEIGHT_NAME).
    3) En cualquier caso, adjuntar nuestro scheduler “blindado”.
    """
    global PIPE
    if PIPE is not None:
        return PIPE

    t0 = time.time()
    print(f"[LOAD] Intentando pipeline directa: {MODEL_ID} (rev={MODEL_REVISION})", flush=True)

    vae = None
    if VAE_ID and AutoencoderKL is not None:
        try:
            vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE, token=HF_TOKEN)
            print("[LOAD] VAE externo cargado", flush=True)
        except Exception as e:
            print(f"[WARN] Fallo cargando VAE '{VAE_ID}': {e}", flush=True)

    pipe = None

    # 1) Intento carga directa del repo (pipeline completa)
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
        print("[LOAD] Pipeline cargada directamente desde MODEL_ID", flush=True)
    except Exception as e1:
        print(f"[LOAD] No es pipeline completa o falló carga directa: {e1}", flush=True)

    # 2) Fallback a SDXL base
    if pipe is None:
        try:
            print(f"[LOAD] Cargando SDXL base: {BASE_SDXL_ID}", flush=True)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                BASE_SDXL_ID,
                torch_dtype=DTYPE,
                vae=vae,
                use_safetensors=True,
                add_watermarker=None,
                token=HF_TOKEN
            )
            print("[LOAD] SDXL base cargada", flush=True)

            # 2a) Intentar aplicar LoRA desde tu repo
            if TRY_LORA:
                try:
                    pipe.load_lora_weights(
                        pretrained_model_name_or_path=MODEL_ID,
                        weight_name=LORA_WEIGHT_NAME,
                        token=HF_TOKEN
                    )
                    pipe.fuse_lora()  # opcional (reduce overhead)
                    print(f"[LOAD] LoRA '{MODEL_ID}/{LORA_WEIGHT_NAME}' aplicada y fusionada", flush=True)
                except Exception as e_lora:
                    print(f"[WARN] No pude aplicar LoRA desde {MODEL_ID}: {e_lora}", flush=True)
        except Exception:
            print("[ERROR] Falla cargando SDXL base:\n" + traceback.format_exc(), flush=True)
            raise

    # Silenciar cosas innecesarias y enviar a GPU
    if hasattr(pipe, "watermark"):   pipe.watermark = None
    if hasattr(pipe, "watermarker"): pipe.watermarker = None
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass

    # Adjuntar siempre nuestro scheduler “blindado”
    _attach_scheduler(pipe)

    pipe.set_progress_bar_config(disable=True)
    pipe.to(DEVICE)

    # xformers opcional
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"[WARN] xformers no habilitado: {e}", flush=True)

    # Pequeños tweaks memoria
    try: pipe.enable_vae_slicing()
    except: pass

    print(f"[LOAD] Modelo listo en {time.time() - t0:.1f}s", flush=True)
    PIPE = pipe
    return PIPE


# ------------------------
# Utilidades de imagen PNG
# ------------------------
def _to_skin_64_png(hi: Image.Image) -> bytes:
    if hi.width != hi.height:
        s = min(hi.width, hi.height)
        hi = hi.crop((0, 0, s, s))
    if hi.mode != "RGBA":
        hi = hi.convert("RGBA")

    scale = max(1, hi.width // 64)
    if scale >= 2:
        lo = Image.new("RGBA", (64, 64))
        src = hi.load()
        dst = lo.load()
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


# -------------
# RunPod handler
# -------------
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

        print(f"[RUN] prompt='{prompt[:120]}' steps={steps} guidance={guidance} "
              f"size={width}x{height} seed={seed}", flush=True)

        pipe = _load_pipe()

        gen = torch.Generator(device=DEVICE)
        if seed is not None:
            gen = gen.manual_seed(int(seed))

        with torch.inference_mode():
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
        return {
            "ok": True,
            "w": 64 if r64 else img.width,
            "h": 64 if r64 else img.height,
            "image_b64": b64
        }

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            msg = ("CUDA OOM: baja GEN_WIDTH/GEN_HEIGHT a 640 o 512, steps≈25, "
                   "y pon Concurrency=1 en el endpoint.")
            print(f"[OOM] {msg}", flush=True)
            return {"ok": False, "error": msg}
        print("[ERROR] RuntimeError:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        print("[ERROR] Exception:\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}


# -------------------------
# Arranque del worker RPOD
# -------------------------
runpod.serverless.start({"handler": handler})


# -----------------
# Test local opcional
# -----------------
if __name__ == "__main__":
    print(handler({"input": {"prompt": "pikachu inspired minecraft skin", "seed": 42}}))