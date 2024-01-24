import numpy as np
import cv2 as cv

nFeatures = 100  # Quantidade de melhores pontos, são ranqueados baseado no seu score. (máxima)
edgeThreshold = (
    1000  # Filtro de borda, quanto maior o valor, menos pontos na borda são filtrados
)
contrastTreshold = 0.09 # Parametro para filtrar pontos em regiões semi uniformes. Maior o valor, menos pontos são produzidos.
sigma = 1.8  # Sigma do gaussiano. Se a imagem estiver borrada é bom diminuir.
octaveLayers = (
    10  # Quantidade de camadas em cada oitava, influencia o contrastThreshold.
)


img = cv.imread("imgs/img01.png")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT.create(nfeatures=nFeatures,
    sigma=sigma,
    edgeThreshold=edgeThreshold,
    contrastThreshold=contrastTreshold,
    nOctaveLayers=octaveLayers,)
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)