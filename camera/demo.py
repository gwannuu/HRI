import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time

from camera import model
from camera import util
from camera.body import Body
# from camera.hand import Hand

from robot_interaction.sync_music_robot_v2 import DanceSystemConfig
DEVICE = DanceSystemConfig.DEVICE

move = 0

body_estimation = Body(device=DEVICE)
# hand_estimation = Hand('model/hand_pose_model.pth')

cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # Set width
# cap.set(4, 480)  # Set height
cap.set(3, 100)  # 프레임 너비를 320으로 설정
cap.set(4, 200)  # 프레임 높이를 240으로 설정

prev_time = 0
prev_candidate = None  # 이전 프레임의 관절 위치 저장
movement_threshold = 3 # 움직임을 판단하는 기준 거리 (픽셀 단위)

movement_history = []  # 움직임 기록
history_length = 10  # 시간 기반 필터링을 위한 기록 길이
movement_count_threshold = 6  # 움직임으로 간주할 최소 기록 개수

while True:
    ret, oriImg = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 신체 관절 정보 추출
    candidate, subset = body_estimation(oriImg)
    # 관절 정보를 콘솔에 출력

    if prev_candidate is not None:
        moved = False
        for i, joint in enumerate(candidate):
            x, y, confidence, _ = joint
            if i < len(prev_candidate):
                prev_x, prev_y, prev_confidence, _ = prev_candidate[i]
                # 현재 관절과 이전 관절의 거리 계산
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if distance > movement_threshold:
                    moved = True
                    break

        # 움직임 기록 추가
        movement_history.append(moved)
        if len(movement_history) > history_length:
            movement_history.pop(0)

        # 시간 기반 필터링
        if movement_history.count(True) >= movement_count_threshold:
            print("움직임 감지!")
            move = 1
        else:
            print("움직임 없음.")
            move = 0 
    else:
        print("첫 프레임 - 움직임 판단 불가.")

    # 이전 프레임의 관절 위치 업데이트
    prev_candidate = copy.deepcopy(candidate)

    # 결과를 그리기
    canvas = copy.deepcopy(oriImg)
    # canvas = util.draw_bodypose(canvas, candidate, subset)

    # FPS 표시
    cv2.putText(canvas, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('demo', canvas)  # Show the video feed with FPS
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
