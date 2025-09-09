from PIL import Image
import io

def to_skin_64_png(highres_img: Image.Image) -> bytes:
    """
    Toma la “skin sheet” de alta resolución (p. ej. 768×768) y la convierte
    en una textura 64×64 apta para Minecraft (pre-1.8/1.8 layout).
    """
    # Validación básica cuadrada
    if highres_img.width != highres_img.height:
        # Algunos pipelines SDXL devuelven 1024 si no se fuerza 768: ajustamos por si acaso
        size = min(highres_img.width, highres_img.height)
        box  = (0, 0, size, size)
        hi   = highres_img.crop(box)
    else:
        hi = highres_img

    # Asegurar RGBA para transparencia de overlay
    if hi.mode != "RGBA":
        hi = hi.convert("RGBA")

    # Downscale con vecino más cercano (evita anti-aliasing)
    skin64 = hi.resize((64, 64), resample=Image.NEAREST)

    # Guardar a PNG bytes
    buf = io.BytesIO()
    skin64.save(buf, format="PNG")
    return buf.getvalue()
