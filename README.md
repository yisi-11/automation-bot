# 🤖 AI Shorts Automation Bot (Auto-Gimbal & YOLOv8)

Este proyecto es un script de Python diseñado para automatizar la creación de contenido vertical (Shorts/Reels/TikTok) a partir de vídeos horizontales largos de YouTube. 

A diferencia de un simple recorte estático, este bot utiliza **Inteligencia Artificial (YOLOv8)** para detectar personas en pantalla y un sistema de **Gimbal Virtual** programado desde cero para seguir la acción suavemente.

## 🚀 Características Principales

* **Descarga Integrada:** Utiliza `yt-dlp` para obtener el material en crudo a máxima calidad de forma automática.
* **Director IA (YOLOv8):** Analiza el vídeo frame a frame identificando sujetos clave (personas, rostros) para generar una trayectoria de cámara.
* **Virtual Gimbal Smoothing:** Algoritmo propio basado en un historial promediado (`collections.deque`) que elimina los movimientos bruscos del recorte, simulando un estabilizador físico.
* **Smart Cropping:** Renderizado optimizado a 1080x1920 con MoviePy, manteniendo al sujeto siempre en el centro del frame (Safe Zones).

## 🛠️ Stack Tecnológico
* **Python 3.11**
* **Ultralytics (YOLOv8):** Visión artificial y Object Tracking.
* **MoviePy / Numpy:** Manipulación de matrices de video y renderizado.
* **yt-dlp:** Interfaz de descarga de video en streaming.

## 🚧 Estado del Proyecto (Beta)
Actualmente el proyecto es un Prototipo Funcional (MVP). 
*Próximos pasos en desarrollo:* - Implementar salto de frames en el análisis de YOLO para reducir la carga de CPU/GPU.
- Añadir interfaz gráfica (GUI) para eliminar parámetros hardcodeados de tiempo.
- Integrar subtitulado automático mediante Whisper AI.
