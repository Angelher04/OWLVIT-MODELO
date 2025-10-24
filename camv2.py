import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- 0. Configuración del dispositivo (CPU o GPU) ---
# Comprueba si hay una GPU compatible con CUDA disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
if not torch.cuda.is_available():
    print("ADVERTENCIA: No se detectó GPU (CUDA). El procesamiento será lento en la CPU.")


# --- 1. Cargar el modelo y procesador (Solo una vez) ---
print("Cargando modelo OwlViT... (esto puede tardar un momento)")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

model.to(device) # <-- MODIFICADO: Mueve el modelo a la GPU
print("Modelo cargado.")

# --- 2. Pedir al usuario las consultas de texto (Solo una vez) ---
user_input = input("\n¿Qué quieres detectar? (separa con comas): ")
text_queries = [query.strip() for query in user_input.split(',')]
texts = [text_queries] 

# --- 3. Iniciar la cámara ---
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("\nIniciando cámara. Presiona 'q' en la ventana de video para salir.")

# --- 4. Bucle principal (procesar fotograma por fotograma) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el fotograma.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # --- 5. Procesar la imagen (en cada fotograma) ---
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # --- INICIO DE LÍNEAS MODIFICADAS ---
    # Mueve todos los tensores de entrada (imagen, texto) a la GPU
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        print(f"Error moviendo tensores a {device}: {e}")
        print("Asegúrate de que tu instalación de PyTorch tenga soporte CUDA.")
        break
    # --- FIN DE LÍNEAS MODIFICADAS ---

    outputs = model(**inputs)

    # --- 6. Post-procesar los resultados ---
    # Los 'outputs' están en la GPU, pero el post-procesador los
    # devolverá ('results') a la CPU para que Pillow/OpenCV los usen.
    target_sizes = torch.Tensor([image.size[::-1]]) 
    
    # Asegúrate de que el threshold (umbral) esté bajo
    results = processor.post_process_object_detection(outputs=outputs, 
                                                  threshold=0.05, 
                                                  target_sizes=target_sizes)
    
    # --- 7. Dibujar los resultados ---
    i = 0 
    text = texts[i]
    # Los 'results' ya están en la CPU, listos para usar
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    draw = ImageDraw.Draw(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        
        labelText = f"{text[label]}: {round(score.item(), 3)}"

        draw.rectangle(box, outline="red", width=3)
        
        text_origin = (box[0], box[1] - 12) 
        text_background = (text_origin[0], text_origin[1], text_origin[0] + len(labelText) * 6, text_origin[1] + 12)
        draw.rectangle(text_background, fill="red")
        draw.text(text_origin, labelText, fill="white")

    # --- 8. Mostrar el resultado ---
    frame_processed_rgb = np.array(image)
    frame_processed = cv2.cvtColor(frame_processed_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow('Deteccion en vivo - Presiona "q" para salir', frame_processed)

    # --- 9. Condición de salida ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 10. Limpieza ---
print("Cerrando cámara.")
cap.release()
cv2.destroyAllWindows()