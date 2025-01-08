import cv2
import numpy as np
import keras
import time
import random
from extract_face import extract_face_features
from detect_face import detect_face
from image_overlay import overlay

# 모델 로드
model = keras.models.load_model('./model/face_emotion.h5')

# 이모지 이미지 로드
image_Angry = cv2.imread('./images/Angry.png', cv2.IMREAD_UNCHANGED)
image_Disgust = cv2.imread('./images/Disgust.png', cv2.IMREAD_UNCHANGED)
image_Fear = cv2.imread('./images/Fear.png', cv2.IMREAD_UNCHANGED)  #이미지가 비어있어요! 경로를 설정해주세요
image_Happy = cv2.imread('./images/Happy.png', cv2.IMREAD_UNCHANGED) #경로 설정이 되었다면 ctrl+s 후 app.py를 실행시켜보세요!
image_Sad = cv2.imread('./images/Sad.png', cv2.IMREAD_UNCHANGED)
image_Surprise = cv2.imread('./images/Surprise.png', cv2.IMREAD_UNCHANGED)
image_Neutral = cv2.imread('./images/Neutral.png', cv2.IMREAD_UNCHANGED)

imgList = [image_Angry, image_Disgust, image_Fear, image_Happy, image_Sad, image_Surprise, image_Neutral]

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    last_prediction_time = time.time()
    last_prediction = None
    frame_count = 0
    target_emotion_change_time = time.time()
    
    # 게임 변수 초기화
    score = 0
    target_emotion = random.randint(0, 6)  # 랜덤 감정 선택
    emotion_match_duration = 0  # 감정 일치 지속 시간
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # 프레임 스킵 (3프레임당 1번만 처리)
        frame_count += 1
        if frame_count % 3 != 0:
            continue
            
        current_time = time.time()
        
        # 5초마다 새로운 타겟 감정 설정
        if current_time - target_emotion_change_time >= 5.0:
            target_emotion = random.randint(0, 6)
            target_emotion_change_time = current_time
            emotion_match_duration = 0
        
        # 1초마다 감정 분석 수행
        if current_time - last_prediction_time >= 1.0:
            try:
                face_index = 0
                gray, detected_faces, coord = detect_face(frame)
                
                if detected_faces is not None and len(detected_faces) > 0:
                    face_zoom = extract_face_features(gray, detected_faces, coord)
                    
                    if face_zoom is not None and len(face_zoom) > 0:
                        face_zoom = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
                        x, y, w, h = coord[face_index]
                        
                        # 감정 예측
                        pred = model.predict(face_zoom, verbose=0)
                        pred_result = np.argmax(pred)
                        last_prediction = (pred, pred_result, x, y, w, h)
                        
                        # 게임 로직: 감정이 일치하는지 확인
                        if pred_result == target_emotion:
                            emotion_match_duration += 1
                            if emotion_match_duration >= 1:  # 1초 이상 유지하면 점수 획득
                                score += 1
                                target_emotion = random.randint(0, 6)  # 새로운 감정 설정
                                target_emotion_change_time = current_time
                                emotion_match_duration = 0
                        else:
                            emotion_match_duration = 0
                            
                        last_prediction_time = current_time
                    else:
                        cv2.putText(frame, "Face not detected", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        last_prediction = None
                else:
                    cv2.putText(frame, "Face not detected", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    last_prediction = None
            
            except Exception as e:
                print(f"Error: {e}")
                cv2.putText(frame, "Face not detected", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                last_prediction = None
                continue
        
        # 게임 정보 표시
        cv2.putText(frame, f"Score: {score}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Target Emotion: {emotions[target_emotion]}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 현재 감정 표시
        if last_prediction is not None:
            pred, pred_result, x, y, w, h = last_prediction
            
            # 현재 감정 텍스트 표시
            current_emotion_text = f"Current: {emotions[pred_result]}"
            cv2.putText(frame, current_emotion_text, 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 오버레이 이미지 표시
            overlay(frame, x + w // 2, y + h // 2, w // 3, h // 3, imgList[pred_result])

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_capture.release()