# handler.py
import os, io, base64, traceback, time
from typing import Any, Dict, List, Tuple
from PIL import Image
import runpod
from gradio_client import Client  # 👈 sin handle_file

HF_SPACE_ID  = os.getenv("HF_SPACE_ID", "phenixrhyder/3D-Minecraft-Skin-Generator")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # opcional (URL directa del Space)
HF_TOKEN     = os.getenv("HF_TOKEN")      # recomendable para evitar rate limit

_client: Client | None = None
_api_name: str | None = None
_param_order: List[str] = []

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _open_any(x: Any) -> Image.Image | None:
    try:
        if isinstance(x, Image.Image):
            return x.convert("RGBA")
        if isinstance(x, (bytes, bytearray)):
            return Image.open(io.BytesIO(x)).convert("RGBA")
        if isinstance(x, tuple) and x:
            return _open_any(x[0])
        if isinstance(x, dict):
            for k in ("path", "name"):
                if isinstance(x.get(k), str):
                    return Image.open(x[k]).convert("RGBA")
        if isinstance(x, str):
            return Image.open(x).convert("RGBA")
    except Exception:
        return None
    return None

def _flatten_images(result: Any) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    def push(v):
        im = _open_any(v)
        if im: imgs.append(im)

    if isinstance(result, list):
        for v in result: push(v)
    else:
        push(result)

    if not imgs and isinstance(result, (list, tuple)) and result:
        maybe = result[0]
        if isinstance(maybe, list):
            for v in maybe: push(v)
        else:
            push(maybe)
    return imgs

def _boot():
    global _client, _api_name, _param_order
    print(f"[boot] connecting to space: {HF_SPACE_ID} (url={HF_SPACE_URL or '-'})")
    _client = Client(HF_SPACE_URL or HF_SPACE_ID, hf_token=HF_TOKEN, verbose=False)

    try:
        info = _client.view_api(return_format="dict")
        named = (info.get("named_endpoints") or {}) if isinstance(info, dict) else {}
        print(f"[boot] endpoints found: {', '.join(named.keys()) or '(none)'}")

        def looks_image_endpoint(spec: Dict[str, Any]) -> bool:
            params = [p.get("name","").lower() for p in (spec.get("parameters") or [])]
            rets = spec.get("returns") or []
            has_prompt = any(n in params for n in ("prompt","text","input","description","caption"))
            returns_image = any(("image" in (r.get("type","")+r.get("label","")).lower()
                                 or "gallery" in (r.get("type","")+r.get("label","")).lower())
                                for r in rets)
            return has_prompt and returns_image

        selected: Tuple[str, Dict[str, Any]] | None = None
        for name, spec in named.items():
            if looks_image_endpoint(spec):
                selected = (name, spec); break
        if not selected and named:
            selected = ("/predict", named.get("/predict") or next(iter(named.items()))[1])

        if selected:
            _api_name = selected[0]
            _param_order = [p.get("name","") for p in (selected[1].get("parameters") or [])]
            print(f"[boot] selected api_name={_api_name} params={_param_order}")
        else:
            _api_name = "/predict"; _param_order = []
            print("[boot] WARNING no named endpoints; fallback to /predict")
    except Exception as e:
        print("[boot] view_api failed:", repr(e))
        _api_name = "/predict"; _param_order = []

_ALIAS = {
    "prompt": ["prompt","text","input","description","caption"],
    "negative_prompt": ["negative_prompt","negativeprompt","neg_prompt","negative"],
    "seed": ["seed","random_seed"],
    "steps": ["steps","num_inference_steps","num_steps"],
    "num_images": ["num_images","num_samples","images","n_images"],
    "guidance_scale": ["guidance_scale","cfg","cfg_scale","guidance"],
}

def _value_from_payload(param_name: str, p: Dict[str, Any]) -> Any:
    pname = param_name.lower()
    if pname in p and p[pname] not in (None, ""):
        return p[pname]
    for canon, aliases in __ALIAS.items():
        if pname == canon or pname in aliases:
            for k in [canon] + aliases:
                v = p.get(k)
                if v not in (None, ""): return v
    if pname == "num_inference_steps" and p.get("steps") not in (None, ""):
        return p["steps"]
    return None

def handler(event: Dict[str, Any]):
    global _client, _api_name, _param_order
    try:
        if _client is None:
            _boot()

        p = (event.get("input") or {}) if isinstance(event, dict) else {}
        if p.get("warmup"):
            return {"status":"WARM","ok":True}

        inputs: List[Any] = []
        if _param_order:
            for param in _param_order:
                inputs.append(_value_from_payload(param, p))

        print(f"[run] api={_api_name} positional={inputs} payload_keys={list(p.keys())}")

        t0 = time.time()
        if _param_order:
            result = _client.predict(*inputs, api_name=_api_name)
        else:
            kwargs = {}
            for canon, aliases in _ALIAS.items():
                for k in [canon] + aliases:
                    v = p.get(k)
                    if v not in (None, ""): kwargs[canon] = v; break
            result = _client.predict(api_name=_api_name, **kwargs)
        dt = time.time() - t0
        print(f"[run] space call done in {dt:.2f}s")

        imgs = _flatten_images(result)
        if not imgs:
            return {"status":"FAILED","error":"Space returned no images"}

        images_b64 = [_to_b64(im) for im in imgs]
        return {
            "status": "COMPLETED",
            "images": images_b64,
            "output": {"images": images_b64},
            "meta": {"elapsed_sec": dt, "count": len(images_b64),
                     "api_name": _api_name, "params": _param_order}
        }
    except Exception as e:
        print("[handler] FAILED:", repr(e))
        traceback.print_exc()
        return {"status":"FAILED","error":str(e)}

runpod.serverless.start({"handler": handler})