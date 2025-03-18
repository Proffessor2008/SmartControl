import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
img_background = cv2.imread("Background.png")
img_game_over = cv2.imread("gameOver.png")
img_ball = cv2.imread("Ball.png", cv2.IMREAD_UNCHANGED)
img_bat1 = cv2.imread("bat1.png", cv2.IMREAD_UNCHANGED)
img_bat2 = cv2.imread("bat2.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ball_pos = [100, 100]
speed_x = 15
speed_y = 15
game_over = False
score = [0, 0]

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img_raw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, img_background, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = img_bat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, img_bat1, (59, y1))
                if 59 < ball_pos[0] < 59 + w1 and y1 < ball_pos[1] < y1 + h1:
                    speed_x = -speed_x
                    ball_pos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, img_bat2, (1195, y1))
                if 1195 - 50 < ball_pos[0] < 1195 and y1 < ball_pos[1] < y1 + h1:
                    speed_x = -speed_x
                    ball_pos[0] -= 30
                    score[1] += 1

    # Game Over
    if ball_pos[0] < 40 or ball_pos[0] > 1200:
        game_over = True

    if game_over:
        img = img_game_over
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)

        # Check for spacebar press to restart
        key = cv2.waitKey(1)
        if key == ord(' '):
            ball_pos = [100, 100]
            speed_x = 15
            speed_y = 15
            game_over = False
            score = [0, 0]
            img_game_over = cv2.imread("Resources/gameOver.png")
    else:
        # Move the Ball
        if ball_pos[1] >= 500 or ball_pos[1] <= 10:
            speed_y = -speed_y

        ball_pos[0] += speed_x
        ball_pos[1] += speed_y

        # Draw the ball
        img = cvzone.overlayPNG(img, img_ball, ball_pos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    img[580:700, 20:233] = cv2.resize(img_raw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ball_pos = [100, 100]
        speed_x = 15
        speed_y = 15
        game_over = False
        score = [0, 0]
