import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def horn_schunck(im1, im2, alpha=1.0, Niter=100):
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    Ix = cv.Sobel(im1, cv.CV_32F, 1, 0, ksize=5)
    Iy = cv.Sobel(im1, cv.CV_32F, 0, 1, ksize=5)
    It = im2 - im1

    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    for _ in range(Niter):
        u_avg = cv.blur(u, (3, 3))
        v_avg = cv.blur(v, (3, 3))

        denom = alpha**2 + Ix**2 + Iy**2

        u = u_avg - Ix * (Ix*u_avg + Iy*v_avg + It) / denom
        v = v_avg - Iy * (Ix*u_avg + Iy*v_avg + It) / denom

    return u, v

im1 = cv.imread('imgs/img01.png', cv.IMREAD_GRAYSCALE)
im2 = cv.imread('imgs/img02.png', cv.IMREAD_GRAYSCALE)

u, v = horn_schunck(im1, im2, alpha=1.0, Niter=100)

flow_magnitude = np.sqrt(u**2 + v**2)
norm_flow_magnitude = cv.normalize(flow_magnitude, None, 0, 255, cv.NORM_MINMAX)

vis_flow_magnitude = norm_flow_magnitude.astype(np.uint8)

plt.imshow(vis_flow_magnitude)
plt.show()