# mapper.py — pasa de render cuadrado a skin sheet 64x64 (passthrough sólido)
from PIL import Image
import io

def _square_rgba(img: Image.Image) -> Image.Image:
    # recorta a cuadrado si hiciera falta y fuerza RGBA
    if img.width != img.height:
        s = min(img.width, img.height)
        img = img.crop((0, 0, s, s))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def to_skin_layout(img: Image.Image) -> bytes:
    """
    El modelo ya produce una 'skin sheet' estilo Minecraft.
    Reducimos a 64x64 con NEAREST (pixel-art nítido) y devolvemos PNG en bytes.
    """
    hi = _square_rgba(img)
    skin64 = hi.resize((64, 64), resample=Image.NEAREST)
    buf = io.BytesIO()
    skin64.save(buf, format="PNG")
    return buf.getvalue()
