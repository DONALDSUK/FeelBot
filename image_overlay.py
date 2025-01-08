import cv2
import numpy as np

def overlay(image, x, y, w, h, overlay_image):
    try:
        # 이미지 경계 확인
        img_h, img_w = image.shape[:2]
        
        # 오버레이할 영역 계산
        x1 = max(x - w, 0)
        y1 = max(y - h, 0)
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        
        # 오버레이 이미지 크기 조정
        overlay_width = x2 - x1
        overlay_height = y2 - y1
        
        if overlay_width <= 0 or overlay_height <= 0:
            return
        
        # overlay_image 크기 조정
        overlay_resized = cv2.resize(overlay_image, (overlay_width, overlay_height))
        
        # 알파 채널 처리
        alpha = overlay_resized[:, :, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        
        # ROI 추출
        roi = image[y1:y2, x1:x2]
        
        # 이미지 합성
        if roi.shape[:2] == overlay_resized.shape[:2]:
            overlay_rgb = overlay_resized[:, :, :3]
            image[y1:y2, x1:x2] = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
            
    except Exception as e:
        print(f"Overlay error: {e}")
        pass