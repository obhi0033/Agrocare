
import cv2, numpy as np
def decode_uploaded_image(uploaded):
    data=np.frombuffer(uploaded.getvalue(),np.uint8)
    return cv2.imdecode(data,cv2.IMREAD_COLOR)
def check_image_quality(img,threshold=100):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    score=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    return {"score":score,"is_sharp":score>=threshold}
