# mapper.py — autocrop de la sheet (quita el preview inferior) + downscale a 64x64
from PIL import Image
import io
from collections import Counter

# Colores de fondo típicos que aparecen en outputs
_BG_CANDIDATES = [
    (255, 0, 255),    # magenta puro
    (128, 128, 128),  # gris medio
    (200, 200, 200),  # gris claro
    (255, 255, 255),  # blanco
    (240, 240, 240),
    (245, 245, 245),
]

def _to_rgba_square(img: Image.Image) -> Image.Image:
    if img.width != img.height:
        s = min(img.width, img.height)
        img = img.crop((0, 0, s, s))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img

def _guess_bg(img: Image.Image):
    """Adivina color de fondo mirando las esquinas y añade candidatos conocidos."""
    px = img.load()
    w, h = img.size
    samples = [
        px[0, 0], px[w-1, 0], px[0, h-1], px[w-1, h-1],
        px[w//2, 0], px[w//2, h-1], px[0, h//2], px[w-1, h//2],
    ]
    # ignorar alpha al elegir fondo
    rgb = [(r, g, b) for (r, g, b, a) in samples]
    most = Counter(rgb).most_common(1)[0][0] if rgb else (255, 255, 255)
    # prioriza el más frecuente pero permite candidatos típicos
    cand = [most] + [c for c in _BG_CANDIDATES if c != most]
    return cand

def _remove_flat_background(img: Image.Image, bg_list, tol: int = 12) -> Image.Image:
    """Hace transparente un fondo casi plano (para que no manche el PNG final)."""
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            for br, bg, bb in bg_list:
                if abs(r - br) <= tol and abs(g - bg) <= tol and abs(b - bb) <= tol:
                    px[x, y] = (r, g, b, 0)
                    break
    return img

def _auto_crop_sheet(img: Image.Image, bg_list, strip_frac_max=0.35, bg_row_ratio=0.85):
    """
    Detecta si hay una franja inferior (preview) y recorta.
    - strip_frac_max: cuánto como máximo puede medir la franja inferior (35%).
    - bg_row_ratio: si una fila es >85% fondo, la consideramos fondo predominante.
    Heurística: subimos desde abajo hasta encontrar zona 'densa' (no fondo),
    esa será la base de la sheet; recortamos arriba-cuadrado desde (0,0).
    """
    px = img.load()
    w, h = img.size

    def is_bg(pixel):
        r, g, b, a = pixel
        if a < 8:
            return True
        for br, bg, bb in bg_list:
            if abs(r - br) <= 14 and abs(g - bg) <= 14 and abs(b - bb) <= 14:
                return True
        return False

    # límite superior de la franja a inspeccionar
    min_y = int(h * (1.0 - strip_frac_max))
    cut_y = h  # por defecto no cortamos

    # desde abajo hacia arriba, buscamos primera fila "densa" (no-fondo)
    for y in range(h - 1, min_y - 1, -1):
        bg_cnt = 0
        for x in range(w):
            if is_bg(px[x, y]):
                bg_cnt += 1
        if (bg_cnt / w) < bg_row_ratio:
            # encontramos zona con contenido: lo tomamos como base de la sheet
            cut_y = y + 1
            break

    # si la franja inferior es grande (p.ej. previews), recortamos al cuadrado desde arriba
    if cut_y < h:
        # tamaño cuadrado posible: min(ancho, cut_y)
        side = min(w, cut_y)
        return img.crop((0, 0, side, side))
    # si no detectamos nada, devolvemos tal cual
    return img

def to_skin_layout(highres_img: Image.Image) -> bytes:
    """
    1) Cuadra y pasa a RGBA.
    2) Elimina fondo plano.
    3) Recorta automáticamente la sheet (quita previews inferiores).
    4) Downscale a 64x64 con vecino más cercano.
    5) Devuelve PNG bytes.
    """
    hi = _to_rgba_square(highres_img)
    bg_list = _guess_bg(hi)
    hi = _remove_flat_background(hi, bg_list, tol=12)
    hi = _auto_crop_sheet(hi, bg_list, strip_frac_max=0.38, bg_row_ratio=0.88)
    skin64 = hi.resize((64, 64), resample=Image.NEAREST)

    buf = io.BytesIO()
    skin64.save(buf, format="PNG")
    return buf.getvalue()
