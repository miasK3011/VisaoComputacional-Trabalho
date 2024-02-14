import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("imgs/dsc02651.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("imgs/dsc02652.jpg", cv.IMREAD_GRAYSCALE)

fast = cv.FastFeatureDetector.create(
    threshold=110, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16, nonmaxSuppression=False
)
kp1 = fast.detect(img, None)
kp2 = fast.detect(img2, None)

img3 = cv.drawKeypoints(img, kp1, None, color=(255, 0, 255), flags=0)
img4 = cv.drawKeypoints(img2, kp2, None, color=(255, 0, 255), flags=0)

print(f"{len(kp1)} pontos")
print(f"{len(kp2)} pontos")
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img3)
plt.title("Imagem 1")  # Opcional: adicione um título à imagem
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img4)
plt.title("Imagem 2")  # Opcional: adicione um título à imagem
plt.axis("off")  # Desativa os eixos para uma visualização mais limpa

plt.show()

# APLICAÇÃO DOS DESCRITORESsift = cv.SIFT.create()
sift = cv.SIFT.create()
_, des1 = sift.compute(img, kp1)
_, des2 = sift.compute(img2, kp2)
bf = cv.BFMatcher.create(cv.NORM_L2, crossCheck=False)

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.60 * n.distance:
        good_matches.append(m)

print(f"{len(good_matches)} boas correspondências encontradas")

img5 = cv.drawMatches(
    img,
    kp1,  
    img2,
    kp2,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
plt.imshow(img5), plt.show()

img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
img1_keypoints = np.copy(img)
img2_keypoints = np.copy(img2)

color_img = (0, 255, 0) 

for match in good_matches:
    pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
    cv.circle(img1_keypoints, pt1, 5, color_img, -1)
    pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
    cv.circle(img2_keypoints, pt2, 5, color_img, -1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_keypoints, cmap='gray')
plt.title("Pontos Correspondentes Imagem 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img2_keypoints, cmap='gray')
plt.title("Pontos Correspondentes Imagem 2")
plt.axis("off")

plt.show()

# DESCRIÇÃO COM ORB
orb = cv.ORB.create()
_, des1_orb = orb.compute(img, kp1)
_, des2_orb = orb.compute(img2, kp2)
bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)
good_matches_orb = []
for m, n in matches_orb:
    if m.distance < 0.50 * n.distance:
        good_matches_orb.append(m)

img6 = cv.drawMatches(
    img,
    kp1,  
    img2,
    kp2,
    good_matches_orb,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

plt.figure(figsize=(10, 5))
plt.imshow(cv.cvtColor(img6, cv.COLOR_BGR2RGB)) 
plt.title("Boas Correspondências com Descritor ORB")
plt.axis("off")
plt.show()