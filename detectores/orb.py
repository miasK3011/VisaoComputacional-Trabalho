import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("imgs/ex1.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("imgs/ex2.png", cv.IMREAD_GRAYSCALE)
orb = cv.ORB.create(nfeatures=50, edgeThreshold=10, scaleFactor=1.1, patchSize=10)
kp1, des1_orb = orb.detectAndCompute(img, None)
kp2, des2_orb = orb.detectAndCompute(img2, None)

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

# APLICAÇÃO DOS DESCRITORES
sift = cv.SIFT.create()

_, des1 = sift.compute(img, keypoints=kp1)
_, des2 = sift.compute(img2, keypoints=kp2)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
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
plt.title("Descritor SIFT"), plt.imshow(img5), plt.show()

# DESCRITOR ORB
bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)
good_matches_orb = []
for m, n in matches_orb:
    if m.distance < 0.75 * n.distance:
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