from diffusers import StableDiffusionPipeline
import torch
# una mujer hermosa blanca con musculos abdomen marcado grandes muslos y ojos azules y con cabello blanco y grandes pies con uñas naturale
#  una mujer hermosa blanca con musculos abdomen marcado grandes muslos y ojos azules y con cabello blanco y grandes pies con uñas naturales a eso sumale que tenga un hombre debajo de sus pies abrazandoselos 
# Entrada del usuario
import sys
import shutil 
import os 
# generame una imagen de un niño de cabello blanco y ojos verdes con aspecto de fantasma pero como una caricatura que su piel parezca una galaxia por los colores que tiene y que sea muy bonito con una capa brillante como la de un principe echa tambien de galaxias
# generame una imagen de un niño de cabello blanco y ojos verdes que se parezca a sevagoth de warframe pero en chiquito

if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
else:
    prompt = input("Escribe lo que deseas generar: ")

def generar(entrada):
    # entrada = input("¿Cómo debe ser tu protectora musculosa? : > ")
    #  hermosa blanca musculosa con grandes muslos y ojos azules tambien cabello blanco y grandes pies
    # Cargar el modelo Counterfeit-V3.0 sin filtro NSFW
    pipe = StableDiffusionPipeline.from_pretrained(
        "gsdf/Counterfeit-V2.5",  # Modelo avanzado estilo waifu/musculoso/nsfw
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None  # Desactiva el filtro NSFW
    )

    # Enviar a CPU (o GPU si tienes)
    pipe.to("cpu")  # Cambia a "cuda" si tienes una GPU compatible

    # Generar imagen
    image = pipe(entrada).images[0]

    # Guardar y mostrar imagen
    nombre_archivo = f"{entrada[:50].replace(' ', '_')}.jpg"
    # Crear carpeta si no existe
    output_dir = r"C:/Users/Usuario/Pictures/ImagenesGeneradasConIa"
    os.makedirs(output_dir, exist_ok=True)
    
    ruta_completa = os.path.join(output_dir, nombre_archivo)
    image.save(ruta_completa)
    image.show()
    print(f"\n✅ Imagen generada guardada como: {nombre_archivo}")

    print(f"\n✅ Imagen guardada como: {nombre_archivo}")
    #shutil.copy(f"{nombre_archivo}", f"C:/modelo_seductor/{nombre_archivo}")
    
generar(prompt)