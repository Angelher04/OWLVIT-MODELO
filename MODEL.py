import requests
from PIL import Image, ImageDraw # <-- Se añade ImageDraw
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

# 1. Cargar el modelo y el procesador
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# 2. Cargar la imagen de ejemplo
url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Definir qué queremos buscar
texts = [["a photo of a cat", "a photo of a dog"]]

# 4. Procesar la imagen y el texto
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# 5. Post-procesar los resultados
target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

# 6. Preparar los resultados para la primera (y única) imagen
i = 0 
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# 7. DIBUJAR LOS RESULTADOS EN LA IMAGEN (Esta es la parte modificada)

# Crear un objeto "dibujable" para la imagen
draw = ImageDraw.Draw(image)

# Iterar sobre cada detección
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    
    # Preparar la etiqueta de texto (ej: "a photo of a cat: 0.98")
    labelText = f"{text[label]}: {round(score.item(), 3)}"

    # Dibujar la caja (bounding box)
    draw.rectangle(box, outline="red", width=3)
    
    # ----- Dibujar la etiqueta con fondo -----
    # Calcular la posición del texto (un poco arriba de la caja)
    text_origin = (box[0], box[1] - 12) 
    
    # Estimar el tamaño del fondo (esto es una aproximación simple)
    text_background = (
        text_origin[0], 
        text_origin[1], 
        text_origin[0] + len(labelText) * 6, # Ajusta el '6' si la fuente es muy grande/pequeña
        text_origin[1] + 12
    )
    
    draw.rectangle(text_background, fill="red")
    # Dibujar el texto
    draw.text(text_origin, labelText, fill="white")
    # ----------------------------------------

    # Imprimir también en la consola (opcional)
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


# 8. Mostrar la imagen con las detecciones
image.show()

# Opcional: Guardar la imagen en disco
# image.save("resultado_deteccion.jpg")