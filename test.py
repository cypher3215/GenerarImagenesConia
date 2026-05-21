from diffusers import StableDiffusionPipeline
import torch

import sys
import shutil 
import os 

if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
else:
    prompt = input("Escribe lo que deseas generar: ")

def generar(entrada):

    pipe = StableDiffusionPipeline.from_pretrained(
        "gsdf/Counterfeit-V2.5",  # Modelo avanzado 
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None 
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

    
generar(prompt)
