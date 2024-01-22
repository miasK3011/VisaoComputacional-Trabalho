# Detectando pontos com ORB e realizando a correspondencia com os descritores SIFT e ORB
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('imgs/exemplo.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('imgs/exemplo2.png', cv.IMREAD_GRAYSCALE)  # trainImage

nFeatures = 20
edgeTreshold = 1
sigma = 0.1

# Initiate SIFT detector
sift = cv.SIFT.create(nfeatures=nFeatures, edgeThreshold=edgeTreshold, sigma=sigma)
orb = cv.ORB.create(nfeatures=nFeatures, edgeThreshold=edgeTreshold, scoreType=1)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Compute descriptors with ORB
_, des1_sift = sift.compute(img1, kp1)
_, des2_sift = sift.compute(img2, kp2)

# BFMatcher with default params
bf = cv.BFMatcher()
matchesSift = bf.knnMatch(des1_sift, des2_sift, k=3)
matchesOrb = bf.knnMatch(des1, des2, k=3)

""" # Apply ratio test
goodOrb = []
for m, n in matchesOrb:
    if m.distance < 0.60 * n.distance:
        goodOrb.append([m])
        
goodSift = []
for m, n in matchesSift:
    if m.distance < 0.60 * n.distance:
        goodSift.append([m])

accuracyOrb = (len(goodOrb)/len(matchesOrb))*100
accuracySIFT = (len(goodSift)/len(matchesSift))*100 """

print("Matches Sift: ", len(matchesSift),"\nMatches Orb: ", len(matchesOrb))
""" print("Good Sift: ", len(goodSift),"\nGood Orb: ", len(goodOrb))
print("Acerto ORB: {:.2f} %".format(accuracyOrb), "\nAcerto Sift: {:.2f} %".format(accuracySIFT)) """

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matchesOrb, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img4 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matchesSift, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.subplot(1, 2, 1)
plt.imshow(img3)
plt.title('Correspondências ORB')

plt.subplot(1, 2, 2)
plt.imshow(img4)
plt.title('Correspondências detector ORB e descritor SIFT')

# Ajuste o layout para evitar sobreposição

# Exiba a janela com os dois subplots
plt.show()