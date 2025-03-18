import time
from ctypes import cast, POINTER

import cv2
import numpy as np
import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import HandFinder as htm

# Camera settings
wCam, hCam = 900, 720
frameR = 100  # Frame Reduction
smoothening = 7  # Smoothening factor for cursor movement

# Initialize variables
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Capture video
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(maxHands=1, detectionCon=0.7)

# Get screen size
wScr, hScr = pyautogui.size()

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
vol, volBar, volPer = 0, 400, 0
area = 0
colorVol = (255, 0, 0)

# Flag to control volume interface visibility
show_volume_interface = True

while True:
    # Capture frame
    success, img = cap.read()

    # Find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # Get coordinates of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

        # Drawing rectangle for cursor movement area
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Cursor Movement Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smooth values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

        # Volume Control Mode
        if show_volume_interface:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            if 250 < area < 1000:
                length, img, lineInfo = detector.findDistance(4, 8, img)

                # Convert length to volume
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])

                # Smooth out the volume changes
                smoothness = 10
                volPer = smoothness * round(volPer / smoothness)

                # Change volume if pinky is down
                if not fingers[4]:
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    colorVol = (0, 255, 0)
                else:
                    colorVol = (255, 0, 0)

                # Draw volume bar
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

                # Display current volume
                cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
                cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Image", img)

    # Check for spacebar press to toggle volume interface
    if cv2.waitKey(1) & 0xFF == ord(' '):
        show_volume_interface = not show_volume_interface

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()