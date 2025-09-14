# handler.py — RunPod: proxy al script oficial de Monadical
import os, io, base64, time, json, subprocess, shlex, tempfile, traceback, fcntl, shutil
import runpod

# --------- Caches HF en volumen persistente ----------
HF_HOME = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)

# Si no está instalado hf_transfer, desactívalo para que no casque
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer  # noqa
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# --------- Paths del repo oficial ----------
REPO_DIR = "/deps/minecraft_skin_generator"
SCRIPT   = os.path.join(REPO_DIR, "bin", "minecraft-skins-sdxl.py")

# --------- Defaults ----------
MODEL_ID       = os.getenv("MODEL_ID", "monadical-labs/minecraft-skin-generator-sdxl")
MODEL_REVISION = os.getenv("MODEL_REVISION", "")  # opcional
GUIDANCE       = float(os.getenv("GUIDANCE", "6.5"))
STEPS          = int(os.getenv("STEPS", "30"))
WIDTH          = int(os.getenv("GEN_WIDTH", "768"))
HEIGHT         = int(os.getenv("GEN_HEIGHT", "768"))
NEG_PROMPT_DEF = os.getenv("NEG_PROMPT", "")
HF_TOKEN       = os.getenv("HUGGING_FACE_HUB_TOKEN", "") or os.getenv("HF_TOKEN", "")

_LOCK_FPATH = "/runpod-volume/mono_init.lock"
os.makedirs(os.path.dirname(_LOCK_FPATH), exist_ok=True)

def _disk_log():
    try:
        total, used, free = shutil.disk_usage("/")
        t2, u2, f2 = shutil.disk_usage("/runpod-volume")
        print(f"[DISK] / free={free/1e9:.1f}GB | /runpod-volume free={f2/1e9:.1f}GB", flush=True)
    except Exception:
        pass

def _ensure_repo():
    if not os.path.exists(SCRIPT):
        raise RuntimeError("Repo Monadical no encontrado dentro de la imagen. Revisa el Dockerfile (git clone).")

def _run_script(prompt, neg, steps, guidance, width, height, seed):
    """
    Llama al script oficial. Devuelve bytes PNG.
    """
    _ensure_repo()
    outdir = tempfile.mkdtemp(prefix="skins-")
    out_png = os.path.join(outdir, "skin.png")

    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    # Construye cmd. El script de Monadical acepta estos flags (prompt, negative, steps, guidance, width, height, seed, output)
    # Si cambian, ajusta aquí tras mirar el script.
    cmd = [
        "python", SCRIPT,
        "--prompt", prompt,
        "--negative", neg or "",
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--width", str(width),
        "--height", str(height),
        "--output", out_png
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if MODEL_ID:
        cmd += ["--model", MODEL_ID]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    print("[CALL]", shlex.join(cmd), flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=REPO_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout generando la skin (script tardó demasiado).")

    print("[SCRIPT OUT]\n" + proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Script salió con código {proc.returncode}")

    if not os.path.exists(out_png):
        raise RuntimeError("El script terminó sin crear el PNG de salida.")

    with open(out_png, "rb") as f:
        return f.read()

def handler(event):
    """
    input:
      prompt (str)               -> obligatorio
      negative_prompt (str, opc) -> por defecto .env NEG_PROMPT
      seed (int, opc)
      steps (int, opc)           -> default STEPS
      guidance (float, opc)      -> default GUIDANCE
      width/height (opc)         -> default 768x768 como la Space
    """
    try:
        _disk_log()
        inp = (event or {}).get("input", {}) or {}
        prompt   = inp.get("prompt", "").strip() or "minecraft skin"
        neg      = inp.get("negative_prompt", NEG_PROMPT_DEF)
        steps    = int(inp.get("steps", STEPS))
        guidance = float(inp.get("guidance", GUIDANCE))
        width    = int(inp.get("width",  WIDTH))
        height   = int(inp.get("height", HEIGHT))
        seed     = inp.get("seed")

        print(f"[RUN] '{prompt[:120]}' steps={steps} guidance={guidance} size={width}x{height} seed={seed}", flush=True)

        png_bytes = _run_script(prompt, neg, steps, guidance, width, height, seed)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"ok": True, "w": 64, "h": 64, "image_b64": b64}
    except Exception as e:
        print("[ERROR]\n" + traceback.format_exc(), flush=True)
        return {"ok": False, "error": str(e)}

runpod.serverless.start({"handler": handler})

# --- Mantener el proceso vivo y loguear salidas “raras”
if __name__ == "__main__":
    import sys, time, traceback
    try:
        # Si ejecutas localmente, sirve como smoke test:
        # print(handler({"input": {"prompt": "test skin", "seed": 0}}))
        # En serverless, start() debe quedarse corriendo; si por lo que sea sale,
        # hacemos loop para no matar el contenedor y ver logs.
        while True:
            time.sleep(3600)
    except SystemExit as e:
        print(f"[FATAL] SystemExit code={e.code}", flush=True)
        traceback.print_exc()
        sys.exit(e.code)
    except Exception:
        print("[FATAL] Unhandled exception at top level:\n" + traceback.format_exc(), flush=True)
        raise
