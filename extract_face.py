import numpy as np
import cv2
from detect_face import shape_x, shape_y

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수  ,, 얼굴 주변 영역을 추가로 포함하기 위한 오프셋 비율(수평, 수직)
def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:
        
        # 얼굴로 감지된 영역
        x, y, w, h = det
        
        # 이미지 경계값 받기,, 여유공간을 추가하여 더 넓은 영역을 포함
        horizontal_offset = int(np.floor(offset_coefficients[0] * w)) # ex) 0.075*200 = 15
        vertical_offset = int(np.floor(offset_coefficients[1] * h))
        
        # gray scacle 에서 해당 위치 가져오기
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        
        # 얼굴 이미지만 확대
        #모든 얼굴 이미지를 동일한 크기로 조정.
        new_extracted_face = cv2.resize(extracted_face, (shape_x, shape_y))

        #정규
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) # sacled
        
        new_face.append(new_extracted_face)
        
    return new_face