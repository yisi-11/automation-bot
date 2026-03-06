import yt_dlp
import os
import shutil
import numpy as np
import time
from moviepy.editor import VideoFileClip, vfx
from ultralytics import YOLO
from collections import deque

# =======================================================
# CLASE: GIMBAL VIRTUAL (ESTABILIZACIÓN)
# =======================================================
class GimbalVirtual:
    def __init__(self, suavidad=30):
        self.history = deque(maxlen=suavidad)
        self.last_valid_center = None

    def actualizar(self, nuevo_objetivo_x, ancho_frame):
        if nuevo_objetivo_x is None:
            target = self.last_valid_center if self.last_valid_center is not None else ancho_frame / 2
        else:
            target = nuevo_objetivo_x
            self.last_valid_center = target
        
        self.history.append(target)
        centro_suave = np.mean(self.history)
        return centro_suave

# Inicialización (solo se ejecuta una vez)
print("🧠 Cargando Director IA (YOLOv8)...")
model = YOLO('yolov8n.pt') 

# =======================================================
# PARTE 1: DESCARGA
# =======================================================
def descargar_video(url, carpeta="videos_originales"):
    if os.path.exists(carpeta):
        try: shutil.rmtree(carpeta); os.makedirs(carpeta)
        except: pass
    else: os.makedirs(carpeta)
    print(f"\n🚀 Descargando material bruto...")
    try:
        ydl_opts = { 'outtmpl': os.path.join(carpeta, '%(id)s.%(ext)s'), 'format': 'best[ext=mp4]/best', 'noplaylist': True, 'quiet': False }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        archivos = [f for f in os.listdir(carpeta) if not f.endswith('.part')]
        if not archivos: return None
        ruta_final = os.path.join(carpeta, "raw_footage.mp4")
        shutil.move(os.path.join(carpeta, archivos[0]), ruta_final)
        return ruta_final
    except Exception as e:
        print(f"❌ Error descarga: {e}")
        return None

# =======================================================
# PASO 1: ANÁLISIS DE LA TRAYECTORIA (YOLO)
# =======================================================
def pre_analizar_trayectoria(clip_sub, fps_analisis=5):
    """Ejecuta YOLO de forma controlada y guarda la posición X de la cámara para cada frame."""
    print("🧠 PASO 1: Analizando y creando guion de cámara...")
    gimbal = GimbalVirtual(suavidad=20)
    trayectoria_x = [] 
    num_frames = int(clip_sub.duration * clip_sub.fps)
    yolo_classes = [0, 39, 41, 63, 67]
    
    for i in range(num_frames):
        t = i / clip_sub.fps
        frame = clip_sub.get_frame(t)
        h_orig, w_orig, _ = frame.shape
        results = model.predict(frame, classes=yolo_classes, verbose=False, imgsz=320)
        
        objetivos_x = []
        for r in results:
            for box in r.boxes:
                x1, _, x2, _ = box.xyxy[0].tolist()
                ancho_obj = x2 - x1
                if ancho_obj > (w_orig * 0.10): 
                    center_x = (x1 + x2) / 2
                    cls = int(box.cls[0])
                    weight = 2.0 if cls == 0 else 1.0 
                    objetivos_x.extend([center_x] * int(weight))

        target_x = np.mean(objetivos_x) if objetivos_x else None
        centro_suave = gimbal.actualizar(target_x, w_orig)
        trayectoria_x.append(centro_suave)
        
        if i % 100 == 0:
            print(f"   Analizado frame {i}/{num_frames}...")

    print("✅ Trayectoria de cámara grabada.")
    return trayectoria_x, w_orig, h_orig, clip_sub.fps

# =======================================================
# PASO 2: EDICIÓN Y RENDERIZADO (MÁXIMA ESTABILIDAD)
# =======================================================
def edicion_profesional(clip_sub, duracion, trayectoria_x, w_orig, h_orig, fps):
    
    # CONFIGURACIÓN TÉCNICA
    W_OUT, H_OUT = 1080, 1920
    
    # NUEVO: Factor de Zoom Estático 1.05x (Punto 4B)
    ZOOM_FACTOR = 1.05
    
    # --- APLICACIÓN DE LA TRAYECTORIA (El Guion) ---
    def crop_inteligente(get_frame, t):
        frame = get_frame(t)
        
        # 1. Buscar la posición X en el guion
        frame_index = int(t * fps)
        
        if frame_index >= len(trayectoria_x):
            center_x = trayectoria_x[-1] 
        else:
            center_x = trayectoria_x[frame_index]
            
        # 2. Calcular el recorte (CROP) en el nuevo tamaño con zoom
        # Reducimos el área de recorte para simular el zoom 1.05x
        crop_width = int((h_orig * (W_OUT / H_OUT)) / ZOOM_FACTOR)
        
        # ZONAS SEGURAS
        x1 = int(center_x - (crop_width / 2))
        x2 = x1 + crop_width
        
        if x1 < 0: x1 = 0; x2 = crop_width
        if x2 > w_orig: x2 = w_orig; x1 = w_orig - crop_width
        
        return frame[:, x1:x2, :]

    # Aplicamos el Recorte Inteligente
    print("\n🎬 PASO 2: Aplicando edición y re-encuadre...")
    clip_cropped = clip_sub.fl(crop_inteligente, apply_to=['mask', 'video'])

    # --- REDIMENSIONADO FINAL SIMPLE ---
    # Solo ajustamos a 1080x1920 (no hay zoom dinámico)
    clip_final = clip_cropped.resize((W_OUT, H_OUT))
    
    # Renderizado
    carpeta_salida = "shorts_finales"
    if not os.path.exists(carpeta_salida): os.makedirs(carpeta_salida)
    ruta_salida = os.path.join(carpeta_salida, "short_estabilizado_final.mp4")

    print(f"⚙️ Renderizando Master (Perfiles Baseline/Simple)...")
    clip_final.write_videofile(
        ruta_salida, fps=30, codec='libx264', audio_codec='aac',
        temp_audiofile='temp-audio.m4a', remove_temp=True, threads=4,
        # CAMBIO CLAVE: Usamos el perfil baseline para máxima estabilidad
        ffmpeg_params=['-pix_fmt', 'yuv420p', '-profile:v', 'baseline'] 
    )
    
    return ruta_salida

# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":
    INICIO = 100 
    DURACION = 60 

    enlace = input("🔗 Pega el enlace del video: ")
    
    if enlace:
        ruta_raw = descargar_video(enlace)
        if ruta_raw:
            clip_original = VideoFileClip(ruta_raw)
            clip_sub = clip_original.subclip(INICIO, INICIO + DURACION)
            
            # --- FLUJO DE DOS PASOS ---
            trayectoria, w_orig, h_orig, fps = pre_analizar_trayectoria(clip_sub)
            
            final = edicion_profesional(clip_sub, DURACION, trayectoria, w_orig, h_orig, fps)
            
            clip_original.close()
            
            if final:
                print(f"\n🌟 ¡PROYECTO FINALIZADO CON MÁXIMA ESTABILIDAD! Archivo: {final}")