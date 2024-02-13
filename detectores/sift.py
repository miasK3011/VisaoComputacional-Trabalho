import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

edgeThreshold = (
    3.5  # Filtro de borda, quanto maior o valor, menos pontos na borda são filtrados
)

img = cv.imread("imgs/dsc02651.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("imgs/dsc02652.jpg", cv.IMREAD_GRAYSCALE)
sift = cv.SIFT.create(
    nfeatures=100,
    edgeThreshold=edgeThreshold,
)

kp1 = sift.detect(img, None)
kp2 = sift.detect(img2, None)

points1 = np.array([k.pt for k in kp1])
points2 = np.array([k.pt for k in kp2])

# Processamento para a primeira imagem
dbscan = DBSCAN(eps=5, min_samples=1).fit(points1)
labels = dbscan.labels_
unique_labels = set(labels)
representative_kp = []
for label in unique_labels:
    index = [i for i, l in enumerate(labels) if l == label]
    centroid = np.mean(points1[index], axis=0)
    best_index = min(
        index,
        key=lambda i: np.linalg.norm(np.array(kp1[i].pt) - centroid),
    )
    best_kp = kp1[best_index]
    representative_kp.append(cv.KeyPoint(x=best_kp.pt[0], y=best_kp.pt[1], size=best_kp.size, angle=best_kp.angle, response=best_kp.response, octave=best_kp.octave, class_id=best_kp.class_id))

# Processamento para a segunda imagem
dbscan = DBSCAN(eps=5, min_samples=1).fit(points2)
labels = dbscan.labels_
unique_labels = set(labels)
representative_kp2 = []
for label in unique_labels:
    index = [i for i, l in enumerate(labels) if l == label]
    centroid = np.mean(points2[index], axis=0)
    best_index = min(
        index,
        key=lambda i: np.linalg.norm(np.array(kp2[i].pt) - centroid),
    )
    best_kp = kp2[best_index]
    representative_kp2.append(cv.KeyPoint(x=best_kp.pt[0], y=best_kp.pt[1], size=best_kp.size, angle=best_kp.angle, response=best_kp.response, octave=best_kp.octave, class_id=best_kp.class_id))

print(f"Tamanho do kp1: {representative_kp.__sizeof__()} bytes")
print(f"Tamanho do kp2: {representative_kp2.__sizeof__()} bytes")

img3 = cv.drawKeypoints(img, representative_kp, None, color=(255, 0, 255), flags=0)
img4 = cv.drawKeypoints(img2, representative_kp2, None, color=(255, 0, 255), flags=0)

print(f"{len(representative_kp)} pontos")
print(f"{len(representative_kp2)} pontos")
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
""" 
# APLICAÇÃO DOS DESCRITORES SIFT
_, des1 = sift.compute(img, representative_kp)
_, des2 = sift.compute(img2, representative_kp2)
bf = cv.BFMatcher.create(cv.NORM_L2, crossCheck=False)

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"{len(good_matches)} boas correspondências encontradas")

img5 = cv.drawMatches(
    img,
    representative_kp,  
    img2,
    representative_kp2,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
plt.title("Descritor SIFT"), plt.imshow(img5), plt.show()

# MOSTRA PONTOS ENCONTRADOS NAS DUAS IMAGENS
img1Color = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
img2Color = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
img1_keypoints = np.copy(img1Color)
img2_keypoints = np.copy(img2Color)

color_img1 = (255, 0, 0)  
color_img2 = (0, 255, 0) 

for match in good_matches:
    pt1 = tuple(np.round(representative_kp[match.queryIdx].pt).astype(int))
    cv.circle(img1_keypoints, pt1, 5, color_img1, -1)
    pt2 = tuple(np.round(representative_kp2[match.trainIdx].pt).astype(int))
    cv.circle(img2_keypoints, pt2, 5, color_img2, -1)

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
 """
# Descritor ORB
orb = cv.ORB.create()
_, des1_orb = orb.compute(img, representative_kp)
_, des2_orb = orb.compute(img2, representative_kp2)

bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)

good_matches_orb = []
for m, n in matches_orb:
    if m.distance < 0.75 * n.distance:
        good_matches_orb.append(m)

img6 = cv.drawMatches(
    img,
    representative_kp,  
    img2,
    representative_kp2,
    good_matches_orb,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

plt.figure(figsize=(10, 5))
plt.imshow(cv.cvtColor(img6, cv.COLOR_BGR2RGB)) 
plt.title("Boas Correspondências com Descritor ORB")
plt.axis("off")
plt.show()
