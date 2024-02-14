import cv2
import numpy as np

def find_common_points(keypoints1, keypoints2, keypoints3, threshold=10):
    common_points = []
    for kp1 in keypoints1:
        for kp2 in keypoints2:
            for kp3 in keypoints3:
                if np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt)) < threshold and np.linalg.norm(np.array(kp1.pt) - np.array(kp3.pt)) < threshold and np.linalg.norm(np.array(kp2.pt) - np.array(kp3.pt)) < threshold:
                    common_points.append(kp1)
    return common_points

image = cv2.imread('imgs/dsc02652.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT.create(nfeatures=80, edgeThreshold=10)
orb = cv2.ORB.create(nfeatures=100, edgeThreshold=2, scaleFactor=1.2, fastThreshold=120)
fast = cv2.FastFeatureDetector.create(threshold=110, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16, nonmaxSuppression=False)

keypoints_sift = sift.detect(gray_image, None)
keypoints_orb = orb.detect(gray_image, None)
keypoints_fast = fast.detect(gray_image, None)

print(f"Quantidade de pontos SIFT: {len(keypoints_sift)}")
print(f"Quantidade de pontos ORB: {len(keypoints_orb)}")
print(f"Quantidade de pontos FAST: {len(keypoints_fast)}")

img_sift = cv2.drawKeypoints(gray_image, keypoints_sift, None, color=(255,0,0))
img_orb = cv2.drawKeypoints(gray_image, keypoints_orb, None, color=(0,255,0))
img_fast = cv2.drawKeypoints(gray_image, keypoints_fast, None, color=(0,0,255))

common_points = find_common_points(keypoints_sift, keypoints_orb, keypoints_fast, threshold=5)
img_common_points = cv2.drawKeypoints(gray_image, common_points, None, color=(0,255,255))

cv2.imshow('SIFT Points', img_sift)
cv2.imshow('ORB Points', img_orb)
cv2.imshow('FAST Points', img_fast)
cv2.imshow('Common Points', img_common_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" cv2.imwrite("resultados/dsc02651SIFT.png", img_sift)
cv2.imwrite("resultados/dsc02651ORB.png", img_orb)
cv2.imwrite("resultados/dsc02651FAST.png", img_fast)
cv2.imwrite("resultados/dsc02651ALL.png", img_common_points) """