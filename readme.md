¡Claro! Aquí tienes un borrador de README.md listo para tu proyecto.

Detector de Objetos Zero-Shot con Cámara Web (OwlViT)
Este script utiliza el modelo OwlViT de Google y la biblioteca Hugging Face Transformers para realizar detección de objetos "zero-shot" en tiempo real.

A diferencia de otros modelos, no necesitas re-entrenarlo. Simplemente ejecutas el script, le dices qué quieres buscar (ej. "a cell phone"), y el modelo lo encontrará en el video de tu cámara web.

El script está configurado para usar una GPU NVIDIA (CUDA) si está disponible, lo cual es altamente recomendado para un rendimiento fluido.

🔧 Instalación y Configuración
Sigue estos pasos para configurar tu entorno.

1. (Opcional) Crear un Entorno Virtual
Se recomienda usar un entorno virtual para evitar conflictos entre bibliotecas.

Bash

# Crear el entorno
python -m venv venv

# Activar en Windows
.\venv\Scripts\activate

# Activar en macOS/Linux
source venv/bin/activate
2. Instalar PyTorch con Soporte GPU (CUDA)
Este es el paso más importante. Para que el script use tu GPU NVIDIA, no puedes instalar PyTorch con un simple pip install torch.

Visita el Sitio Web Oficial de PyTorch.

Usa el generador de comandos. Selecciona las opciones adecuadas:

PyTorch Build: Stable

Your OS: Windows (o el que uses)

Package: Pip

Language: Python

Compute Platform: CUDA 11.8 o CUDA 12.1 (Elige la más reciente compatible)

Copia y ejecuta el comando que te proporcionan. Será similar a este:

Bash

# ¡USA EL COMANDO DE LA PÁGINA OFICIAL! Este es solo un ejemplo:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
3. Instalar las demás dependencias
Una vez instalado PyTorch, instala el resto de bibliotecas necesarias:

Bash

pip install transformers opencv-python pillow
▶️ Ejecución
Guarda el código en un archivo (ej. detect_webcam.py).

Ejecuta el script desde tu terminal (asegúrate de que tu entorno virtual esté activado):

Bash

python detect_webcam.py
El script cargará el modelo. Cuando esté listo, te preguntará en la consola: ¿Qué quieres detectar? (separa con comas):

Escribe los objetos que quieres buscar, en inglés y separados por comas.

Ejemplo: a person, a cell phone, a water bottle

Se abrirá una ventana de OpenCV mostrando tu cámara. Los objetos detectados tendrán un recuadro rojo.

Para salir, presiona la tecla 'q' en la ventana de video.

💡 Funcionamiento y Posibles Desafíos
Cómo Funciona
El script captura un fotograma de la cámara, lo convierte a un formato de imagen PIL, y lo pasa al Procesador de OwlViT junto con tus consultas de texto. El Modelo (que se ejecuta en la GPU) procesa esta información y genera puntuaciones y coordenadas (bounding boxes). Finalmente, el script dibuja estas cajas sobre el fotograma y lo muestra en la pantalla.

Desafíos Comunes (Troubleshooting)
Desafío: El script dice Usando dispositivo: cpu o ADVERTENCIA: No se detectó GPU (CUDA).

Causa: PyTorch no está instalado con soporte CUDA (ver Causa 2 de la instalación) o tus controladores (drivers) de NVIDIA están desactualizados.

Solución: Desinstalar torch (pip uninstall torch) y reinstalarlo usando el comando correcto del sitio web de PyTorch.

Desafío: El video es muy lento, tiene "lag" o está entrecortado.

Causa: El modelo se está ejecutando en la CPU, la cual no es lo suficientemente rápida para esta tarea en tiempo real.

Solución: Solucionar el problema de detección de CUDA (ver punto anterior).

Desafío: No se detecta nada, aunque el objeto esté claramente en la cámara.

Causa 1 (Texto): El texto de búsqueda (prompt) no es ideal. El modelo fue entrenado con leyendas descriptivas.

Solución 1: Sé más descriptivo. En lugar de person, usa a person o a photo of a person. En lugar de telefono, usa a cell phone.

Causa 2 (Umbral): El modelo sí detecta el objeto, pero con una confianza baja (ej. 8%), y el código lo está filtrando.

Solución 2: Asegúrate de que el threshold (umbral) en el código esté bajo. En el script lo ajustamos a 0.05 (5% de confianza), lo cual es bueno para video.
