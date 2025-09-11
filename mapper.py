# mapper.py — limpieza de fondo + downscale 64x64 con NEAREST
from PIL import Image
import io, collections

def _most_common_color(img: Image.Image) -> tuple:
    small = img.resize((64, 64), Image.NEAREST).convert("RGBA")
    cnt = collections.Counter(small.getdata())
    cnt.pop((0,0,0,0), None)
    return cnt.most_common(1)[0][0] if cnt else (0, 0, 0, 255)

def _make_bg_transparent(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = _most_common_color(img)
    px = img.load()
    w, h = img.size
    tol = 18
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if (abs(r - bg[0]) <= tol and abs(g - bg[1]) <= tol and abs(b - bg[2]) <= tol):
                px[x, y] = (r, g, b, 0)
    return img

def to_skin_64_png(highres_img: Image.Image) -> bytes:
    if highres_img.width != highres_img.height:
        s = min(highres_img.width, highres_img.height)
        highres_img = highres_img.crop((0, 0, s, s))
    hi = _make_bg_transparent(highres_img)
    skin64 = hi.resize((64, 64), resample=Image.NEAREST).convert("RGBA")
    buf = io.BytesIO()
    skin64.save(buf, format="PNG")
    return buf.getvalue()
