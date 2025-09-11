# mapper.py — convierte un render cuadrado en skin layout 64x64
from PIL import Image

def to_skin_layout(img: Image.Image) -> Image.Image:
    """
    Convierte una imagen cuadrada (ej. 768x768) en un layout de skin de Minecraft 64x64 (pre-1.8).
    Layout oficial: https://minecraft.fandom.com/wiki/Skin#Skin_file
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    skin = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    part = img.resize((16, 16), Image.NEAREST)

    # cabeza
    skin.paste(part, (8, 0))    # frente
    skin.paste(part, (0, 0))    # izquierda
    skin.paste(part, (16, 0))   # derecha
    skin.paste(part, (24, 0))   # atrás
    skin.paste(part, (8, 8))    # arriba
    skin.paste(part, (16, 8))   # abajo

    # cuerpo
    skin.paste(part, (20, 16))  # frente
    skin.paste(part, (28, 16))  # derecha
    skin.paste(part, (32, 16))  # atrás
    skin.paste(part, (16, 16))  # izquierda
    skin.paste(part, (20, 20))  # arriba
    skin.paste(part, (28, 20))  # abajo

    # brazos
    skin.paste(part, (44, 16))  # brazo derecho frente
    skin.paste(part, (40, 16))  # brazo derecho izquierda
    skin.paste(part, (48, 16))  # brazo derecho atrás
    skin.paste(part, (44, 20))  # brazo derecho arriba
    skin.paste(part, (48, 20))  # brazo derecho abajo

    skin.paste(part, (36, 48))  # brazo izquierdo frente
    skin.paste(part, (32, 48))  # brazo izquierdo izquierda
    skin.paste(part, (40, 48))  # brazo izquierdo atrás
    skin.paste(part, (36, 52))  # brazo izquierdo arriba
    skin.paste(part, (40, 52))  # brazo izquierdo abajo

    # piernas
    skin.paste(part, (4, 16))   # pierna derecha frente
    skin.paste(part, (0, 16))   # pierna derecha izquierda
    skin.paste(part, (8, 16))   # pierna derecha atrás
    skin.paste(part, (4, 20))   # pierna derecha arriba
    skin.paste(part, (8, 20))   # pierna derecha abajo

    skin.paste(part, (20, 48))  # pierna izquierda frente
    skin.paste(part, (16, 48))  # pierna izquierda izquierda
    skin.paste(part, (24, 48))  # pierna izquierda atrás
    skin.paste(part, (20, 52))  # pierna izquierda arriba
    skin.paste(part, (24, 52))  # pierna izquierda abajo

    return skin