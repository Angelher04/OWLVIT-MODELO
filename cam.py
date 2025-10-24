import cv2  # Importamos OpenCV
import numpy as np
from PIL import Image, ImageDraw
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- 1. Cargar el modelo y procesador (Solo una vez) ---
print("Cargando modelo OwlViT... (esto puede tardar un momento)")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
print("Modelo cargado.")

# --- 2. Pedir al usuario las consultas de texto (Solo una vez) ---
user_input = input("\n¿Qué quieres detectar? (separa con comas): ")
# Ejemplo: a person, a cell phone, a water bottle

# Procesamos la entrada del usuario para que coincida con el formato de OwlViT
text_queries = [query.strip() for query in user_input.split(',')]
texts = [text_queries] # Formato requerido: [["query1", "query2", ...]]

# --- 3. Iniciar la cámara ---
cap = cv2.VideoCapture(0) # 0 es la cámara web por defecto

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("\nIniciando cámara. Presiona 'q' en la ventana de video para salir.")

# --- 4. Bucle principal (procesar fotograma por fotograma) ---
while True:
    # Capturar un fotograma
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el fotograma.")
        break

    # Convertir el fotograma de OpenCV (BGR) a PIL (RGB)
    # El modelo OwlViT espera una imagen en formato PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # --- 5. Procesar la imagen (en cada fotograma) ---
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # --- 6. Post-procesar los resultados ---
    # Usamos el tamaño de la imagen PIL
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, 
                                                threshold=0.05,  # <-- ¡CAMBIA ESTO!
                                                target_sizes=target_sizes)
    # --- 7. Dibujar los resultados ---
    i = 0 # Solo procesamos una imagen a la vez
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Creamos un objeto ImageDraw para dibujar sobre la imagen PIL
    draw = ImageDraw.Draw(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        
        # Omitimos la parte de "a photo of a" para que la etiqueta sea más limpia
        labelText = f"{text[label]}: {round(score.item(), 3)}"

        # Dibujar la caja
        draw.rectangle(box, outline="red", width=3)
        
        # Dibujar fondo para el texto
        text_origin = (box[0], box[1] - 12) 
        text_background = (text_origin[0], text_origin[1], text_origin[0] + len(labelText) * 6, text_origin[1] + 12)
        draw.rectangle(text_background, fill="red")
        draw.text(text_origin, labelText, fill="white")

    # --- 8. Mostrar el resultado ---
    # Convertir la imagen PIL (con dibujos) de vuelta a formato OpenCV (BGR)
    frame_processed_rgb = np.array(image)
    frame_processed = cv2.cvtColor(frame_processed_rgb, cv2.COLOR_RGB2BGR)

    # Mostrar el fotograma en una ventana
    cv2.imshow('Deteccion en vivo - Presiona "q" para salir', frame_processed)

    # --- 9. Condición de salida ---
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 10. Limpieza ---
print("Cerrando cámara.")
cap.release()
cv2.destroyAllWindows()
