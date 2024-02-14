import cv2
import numpy as np

img1 = cv2.imread("imgs/dsc02651.jpg")
img2 = cv2.imread("imgs/dsc02652.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector.create(threshold=110, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16, nonmaxSuppression=False)

keypoints = fast.detect(gray1, None)
print(len(keypoints))

points1 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None)

good_points1 = points1[status == 1]
good_points2 = points2[status == 1]

# Adicionando filtragem baseada na magnitude do movimento
min_dist = 2
max_dist = 101
filtered_points1 = []
filtered_points2 = []

for i, (new, old) in enumerate(zip(good_points2, good_points1)):
    dist = np.linalg.norm(new - old)
    if min_dist <= dist <= max_dist:
        filtered_points1.append(old)
        filtered_points2.append(new)

for new, old in zip(filtered_points2, filtered_points1):
    a, b = new.ravel()
    c, d = old.ravel()
    a, b, c, d = int(a), int(b), int(c), int(d)
    
    img2 = cv2.circle(img2, (a, b), 2, (0, 0, 255), -1)
    img1 = cv2.circle(img1, (c, d), 2, (0, 0, 255), -1)
    img2 = cv2.line(img2, (a, b), (c, d), (0, 0, 255), 1)

cv2.imshow('Fluxo Optico Filtrado', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("resultadosFluxoOptico/lucasKanade/dsc02651/FAST - Fluxo optico.png", img2)