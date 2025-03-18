import math
import time

import cv2
import mediapipe as mp
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=6, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def save_canvas(canvas, filename="drawing.png"):
    cv2.imwrite(filename, canvas)
    print(f"Рисунок сохранен как {filename}")


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    # Получаем размеры изображения с камеры
    success, img = cap.read()
    if not success:
        print("Не удалось получить изображение с камеры.")
        return
    h, w, c = img.shape

    # Увеличиваем размер холста в 2 раза
    canvas_height = h * 2
    canvas_width = w * 2

    # Создаем белый холст
    canvas = np.ones((canvas_height, canvas_width, 3), np.uint8) * 255  # Белый холст
    current_color = (0, 0, 0)  # Начальный цвет кисти (синий)
    prev_point = None
    eraser_mode = False  # Режим ластика
    drawing = False  # Флаг для определения, нужно ли рисовать

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            # Получаем координаты указательного и среднего пальцев
            index_finger = lmList[8][1], lmList[8][2]
            middle_finger = lmList[12][1], lmList[12][2]

            # Вычисляем расстояние между указательным и средним пальцами
            distance = math.hypot(index_finger[0] - middle_finger[0], index_finger[1] - middle_finger[1])

            # Если расстояние меньше порога (например, 30 пикселей), не рисуем
            if distance < 37:
                drawing = False
            else:
                drawing = True

            # Если рисование разрешено, рисуем
            if drawing:
                if prev_point is None:
                    prev_point = index_finger

                # Если включен режим ластика, рисуем белым цветом
                if eraser_mode:
                    cv2.line(canvas, prev_point, index_finger, (255, 255, 255), 20)  # Ластик
                else:
                    cv2.line(canvas, prev_point, index_finger, current_color, 5)  # Кисть

                prev_point = index_finger
            else:
                prev_point = None
        else:
            prev_point = None

        # Отображаем холст
        cv2.imshow("Canvas", canvas)

        # Создаем кнопки для выбора цвета и ластика
        button_height = 50
        button_width = 100
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, color in enumerate(colors):
            cv2.rectangle(img, (i * button_width, 0), ((i + 1) * button_width, button_height), color, -1)
            cv2.putText(img, f"Color {i + 1}", (i * button_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

        # Кнопка для ластика
        eraser_button_pos = (len(colors) * button_width, 0), ((len(colors) + 1) * button_width, button_height)
        cv2.rectangle(img, eraser_button_pos[0], eraser_button_pos[1], (128, 128, 128), -1)
        cv2.putText(img, "Eraser", (eraser_button_pos[0][0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        # Кнопка для сохранения рисунка (внизу окна)
        save_button_pos = (0, h - button_height), (button_width, h)
        cv2.rectangle(img, save_button_pos[0], save_button_pos[1], (0, 255, 255), -1)
        cv2.putText(img, "Save", (save_button_pos[0][0] + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        # Проверяем, нажата ли кнопка выбора цвета или ластика
        if len(lmList) != 0:
            for i, color in enumerate(colors):
                if (i * button_width < lmList[8][1] < (i + 1) * button_width) and (
                        0 < lmList[8][2] < button_height):
                    current_color = color
                    eraser_mode = False  # Выключаем ластик при выборе цвета

            # Проверяем, нажата ли кнопка ластика
            if (eraser_button_pos[0][0] < lmList[8][1] < eraser_button_pos[1][0]) and (
                    0 < lmList[8][2] < button_height):
                eraser_mode = True  # Включаем ластик

            # Проверяем, нажата ли кнопка сохранения
            if (save_button_pos[0][0] < lmList[8][1] < save_button_pos[1][0]) and (
                    h - button_height < lmList[8][2] < h):
                save_canvas(canvas)  # Сохраняем рисунок

        # Обрабатываем нажатие пробела для очистки полотна
        key = cv2.waitKey(1)
        if key == 32:  # 32 - это код пробела
            canvas = np.ones((canvas_height, canvas_width, 3), np.uint8) * 255  # Очищаем полотно

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Выход по нажатию Esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
