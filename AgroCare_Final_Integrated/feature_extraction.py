
import cv2, numpy as np
def extract_features(img):
    im=cv2.resize(img,(256,256))
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    h,s,v=cv2.split(hsv)
    green=((h>=35)&(h<=90)&(s>40)).mean()
    yellowbrown=((h>=8)&(h<=35)&(s>40)).mean()
    dark=(v<80).mean()
    edges=cv2.Canny(gray,80,160)
    return [h.mean(),h.std(),s.mean(),s.std(),v.mean(),v.std(),green,yellowbrown,dark,(edges>0).mean(),gray.std()]
