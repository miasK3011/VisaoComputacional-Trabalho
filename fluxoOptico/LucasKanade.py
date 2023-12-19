import numpy as np
import cv2
import os

# Parâmetros para detecção de features
feature_params = dict(maxCorners=150, qualityLevel=0.3, minDistance=10, blockSize=10)

# Parâmetros para o Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Cores aleatórias
color = np.random.randint(0, 255, (150, 3))

# Diretório das imagens
dir_imagens = 'fluxoOptico/imgs'

# Ordenar os nomes dos arquivos
arquivos = sorted(os.listdir(dir_imagens))

# Inicializar a primeira imagem
old_frame = cv2.imread(os.path.join(dir_imagens, arquivos[0]))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Loop sobre as imagens restantes
for arquivo in arquivos[1:]:
    new_frame = cv2.imread(os.path.join(dir_imagens, arquivo))
    frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # Calcular o fluxo óptico
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Selecionar somente as melhores features
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Desenhar o fluxo óptico
    mask = np.zeros_like(old_frame)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    # Mostrar a imagem
    cv2.imshow('Fluxo Óptico', img)
    cv2.waitKey(0)

    # Atualizar a imagem antiga e as features
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
