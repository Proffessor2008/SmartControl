import threading
import tkinter as tk
from tkinter import filedialog

import cv2
from cvzone.HandTrackingModule import HandDetector
from ursina import *

# Глобальные переменные для обмена данными между потоками
scale = 0
cx, cy = 0, 0
rotation_y = 0
update_data = False
gesture_mode = 'scale'  # 'scale' или 'rotate'
data_lock = threading.Lock()  # Блокировка для безопасного доступа к данным в нескольких потоках


def process_video():
    global scale, cx, cy, rotation_y, update_data, gesture_mode
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.7)
    startDist = None
    startAngle = None

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)

        if len(hands) == 2:
            hand1, hand2 = hands

            # Жест для масштабирования (два указательных и больших пальца)
            if detector.fingersUp(hand1) == [1, 1, 0, 0, 0] and detector.fingersUp(hand2) == [1, 1, 0, 0, 0]:
                gesture_mode = 'scale'
                if startDist is None:
                    length, info, img = detector.findDistance(hand1["center"], hand2["center"], img)
                    startDist = length

                length, info, img = detector.findDistance(hand1["center"], hand2["center"], img)
                with data_lock:
                    scale = int((length - startDist) // 2)
                    cx, cy = info[4:]
                    update_data = True

            # Жест для вращения (все пальцы подняты)
            elif detector.fingersUp(hand1) == [1, 1, 1, 1, 1] and detector.fingersUp(hand2) == [1, 1, 1, 1, 1]:
                gesture_mode = 'rotate'

                # Вычисляем угол между руками
                angle = math.degrees(math.atan2(hand2["center"][1] - hand1["center"][1],
                                                hand2["center"][0] - hand1["center"][0]))

                if startAngle is None:
                    startAngle = angle

                # Вычисляем изменение угла
                with data_lock:
                    rotation_y = angle - startAngle
                    update_data = True

        else:
            startDist = None
            startAngle = None

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def choose_model_file():
    root = tk.Tk()
    root.withdraw()  # Скрываем основное окно tkinter
    file_path = filedialog.askopenfilename(
        title="Выберите файл модели",
        filetypes=(("GLB files", "*.glb"), ("All files", "*.*"))

    )
    file_path = file_path.split('/')
    return file_path[-1]


# Запуск обработки видео в отдельном потоке
video_thread = threading.Thread(target=process_video)
video_thread.daemon = True
video_thread.start()

# Инициализация Ursina
app = Ursina()

# Настройка сцены
EditorCamera()
model = Entity(model=choose_model_file(), scale=1)
model.position = Vec3(0, 0, 0)  # Позиция модели

# Начальная позиция камеры
camera.position = Vec3(0, 0, -10)  # Камера смотрит на модель с расстояния


def update():
    global scale, cx, cy, rotation_y, update_data, gesture_mode

    with data_lock:
        if update_data:
            if gesture_mode == 'scale':
                # Ограничение масштабирования
                scale_factor = max(0.1, min(5, 1 + scale / 100))  # Ограничиваем масштаб от 0.1 до 5
                model.scale = scale_factor

            elif gesture_mode == 'rotate':
                # Обновление вращения
                if not math.isnan(rotation_y) and not math.isinf(rotation_y):
                    model.rotation_y = rotation_y

            update_data = False


app.run()
