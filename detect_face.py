import cv2

shape_x = 48
shape_y = 48

# 전체 이미지에서 얼굴을 찾아내는 함수
def detect_face(frame):
    
    # cascade pre-trained 모델 불러오기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # RGB를 gray scale로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cascade 멀티스케일 분류 detectMultiScale 반환값  = 얼굴 위치를 나타내는 리스트 x,y,w,h
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor = 1.1, #원래크기 10%씩 축소하면서 탐지
                                                   minNeighbors = 6, # 후보 영역이 얼굴로 인정되기 위한 최소 이웃 개수
                                                   minSize = (shape_x, shape_y), #탐지할 얼굴의 최소 크기
                                                   flags = cv2.CASCADE_SCALE_IMAGE #Haar Cascade의 스케일 방식 설정
                                                  )
    
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y+h, x:x+w] #얼굴 영역 추출
            coord.append([x, y, w, h]) #탐지된 얼굴 좌표 추가
            
    return gray, detected_faces, coord