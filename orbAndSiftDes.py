# Detectando pontos com ORB e SIFT e seus desccritores
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

GOOD_THRESHOLD = 0.60

img1 = cv.imread('imgs/exe1.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('imgs/exe2.jpg', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate detectors
sift = cv.SIFT().create()
fast = cv.FastFeatureDetector().create()
orb = cv.ORB().create()

# FAST com SIFT e ORB
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)
_ , des1_s = sift.compute(img1, kp1)
_ , des2_s = sift.compute(img2, kp2)
_ , des1_o = orb.compute(img1, kp1)
_ , des2_o = orb.compute(img2, kp2)

# Compute descriptors with ORB
kp1S, des1_sift = sift.detectAndCompute(img1, None)
kp2S, des2_sift = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matchesSift = bf.knnMatch(des1_sift, des2_sift, k=2)
matchesFastSift = bf.knnMatch(des1_s, des2_s, k=2)
matchesFastOrb = bf.knnMatch(des1_o, des2_o, k=2)

# Apply ratio test
goodFastSift = []
for m, n in matchesFastSift:
    if m.distance < GOOD_THRESHOLD * n.distance:
        goodFastSift.append([m])

goodFastOrb = []
for m, n in matchesFastOrb:
    if m.distance < GOOD_THRESHOLD * n.distance:
        goodFastOrb.append([m])
        
goodSift = []
for m, n in matchesSift:
    if m.distance < GOOD_THRESHOLD * n.distance:
        goodSift.append([m])

accuracyFastSift = (len(goodFastSift)/len(matchesFastSift))*100
accuracyFastOrb = (len(goodFastOrb)/len(matchesFastOrb))*100
accuracySIFT = (len(goodSift)/len(matchesSift))*100


print("Pontos detectados Fast:\nPrimeira Imagem: {}\nSegunda Imagem: {}\n".format(len(kp1), len(kp2)))
print("Pontos detectados Sift:\nPrimeira Imagem: {}\nSegunda Imagem: {}\n".format(len(kp1S), len(kp2S)))
print(40*"=")
print("Matches Sift: ", len(matchesSift),"\nMatches Fast-Sift: ", len(matchesFastSift), "\nMatches Fast-Orb: ", len(matchesFastOrb))
print("Good Sift: ", len(goodSift),"\nGood Fast-Sift: ", len(goodFastSift),"\nGood Fast-Orb: ", len(goodFastOrb))
print("Acerto Fast-Sift: {:.2f} %".format(accuracyFastSift), "\nAcerto Fast-Orb: {:.2f} %".format(accuracyFastOrb), "\nAcerto Sift: {:.2f} %".format(accuracySIFT))

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, goodFastSift[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img4 = cv.drawMatchesKnn(img1, kp1, img2, kp2, goodFastOrb[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img5 = cv.drawMatchesKnn(img1, kp1S, img2, kp2S, goodSift[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure("Fast Detector- SIFT Descriptor")
plt.imshow(img3)
plt.title('Correspondências Fast-Sift')

plt.figure("Fast Detector- Orb Descriptor")
plt.imshow(img4)
plt.title('Correspondências Fast-Orb')

plt.figure("SIFT Detector-Descriptor")
plt.imshow(img5)
plt.title('Correspondências SIFT')

plt.show()