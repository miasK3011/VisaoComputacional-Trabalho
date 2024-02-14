import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

edgeThreshold = (
    10  # Filtro de borda, quanto maior o valor, menos pontos na borda são filtrados
)

img = cv.imread("imgs/dsc02651.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("imgs/dsc02652.jpg", cv.IMREAD_GRAYSCALE)
sift = cv.SIFT.create(
    nfeatures=80,
    edgeThreshold=edgeThreshold,
)

kp1, des1_sift = sift.detectAndCompute(img, None)
kp2, des1_sift = sift.detectAndCompute(img2, None)

img3 = cv.drawKeypoints(img, kp1, None, color=(255, 0, 255), flags=0)
img4 = cv.drawKeypoints(img2, kp2, None, color=(255, 0, 255), flags=0)

print(f"{len(kp1)} pontos")
print(f"{len(kp2)} pontos")
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img3)
plt.title("Imagem 1")  
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img4)
plt.title("Imagem 2")  
plt.axis("off") 

plt.show()

# APLICAÇÃO DOS DESCRITORES SIFT
_, des1 = sift.compute(img, kp1)
_, des2 = sift.compute(img2, kp2)
bf = cv.BFMatcher.create(cv.NORM_L2, crossCheck=False)

matches = bf.knnMatch(des1, des2, k=2)

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