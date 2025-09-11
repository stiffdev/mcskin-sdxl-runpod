# mapper.py
from PIL import Image
import io

def _square_rgba(img: Image.Image) -> Image.Image:
    # Recorta a cuadrado y asegura RGBA
    if img.width != img.height:
        s = min(img.width, img.height)
        img = img.crop((0, 0, s, s))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def to_skin_64_png(highres_img: Image.Image) -> bytes:
    """
    Caso correcto: el modelo ya genera una 'skin sheet' (layout tipo Minecraft)
    en alta resolución. Solo la reducimos a 64x64 con vecino más cercano.
    """
    hi = _square_rgba(highres_img)
    skin64 = hi.resize((64, 64), resample=Image.NEAREST)
    buf = io.BytesIO()
    skin64.save(buf, format="PNG")
    return buf.getvalue()

# Alias para el handler (importa to_skin_layout)
def to_skin_layout(highres_img: Image.Image) -> bytes:
    return to_skin_64_png(highres_img)
