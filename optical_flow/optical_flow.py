import cv2
import numpy as np
import matplotlib.pyplot as plt


# Gradient
def get_grad(img1, img2):
    kernel = np.array([-1, 8, 0, -8, 1]) / 12
    kernel = kernel.reshape(1, 5)
    Ix = cv2.filter2D(img1, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.filter2D(img1, ddepth=cv2.CV_32F, kernel=kernel.T, borderType=cv2.BORDER_REFLECT)
    It = img2 - img1

    return Ix, Iy, It


# Lambda and optical flow calculation
def optical_flow(ix, iy, it):
    # Gaussian mask
    mask = np.array([1, 4, 6, 4, 1]) / 16
    mask = mask.reshape(1, 5)
    w = mask.T @ mask

    # Getting Ix^2, Iy^2, IxIy, IxIt, IyIt, w^2
    ix_squared = ix * ix
    iy_squared = iy * iy
    IxIy = ix * iy
    IxIt = ix * it
    IyIt = iy * it
    w_squared = w * w

    # Components a, b and c of matrix S
    a = cv2.filter2D(ix_squared, ddepth=cv2.CV_32F, kernel=w_squared, borderType=cv2.BORDER_REFLECT)
    b = cv2.filter2D(IxIy, ddepth=cv2.CV_32F, kernel=w_squared, borderType=cv2.BORDER_REFLECT)
    c = cv2.filter2D(iy_squared, ddepth=cv2.CV_32F, kernel=w_squared, borderType=cv2.BORDER_REFLECT)

    # w^2 * IxIt and w^2 * IyIt
    xt = cv2.filter2D(IxIt, ddepth=cv2.CV_32F, kernel=w_squared, borderType=cv2.BORDER_REFLECT)
    yt = cv2.filter2D(IyIt, ddepth=cv2.CV_32F, kernel=w_squared, borderType=cv2.BORDER_REFLECT)

    # Lambda 1 and lambda 2
    lambd1 = ((a + c) / 2) + np.sqrt((4 * b * b) + (a - c) ** 2) / 2
    lambd2 = ((a + c) / 2) - np.sqrt((4 * b * b) + (a - c) ** 2) / 2

    # Calculating (u, v)
    thresholds = [0.0001, 0.0039, 0.009, 1]
    u = np.empty(lambd1.shape)
    v = np.empty(lambd1.shape)
    for t in thresholds:
        for i in range(lambd1.shape[0]):
            for j in range(lambd1.shape[1]):
                if lambd1[i, j] >= t and lambd2[i, j] >= t:
                    uv = np.linalg.inv(np.array([[a[i, j], b[i, j]],
                                                [b[i, j], c[i, j]]]).reshape(2, 2)) @ np.array([-xt[i, j], -yt[i, j]]).reshape(2, 1)

                    u[i, j] = uv[0]
                    v[i, j] = uv[1]

                elif lambd1[i, j] >= t and lambd2[i, j] < t:
                    if np.linalg.norm([ix[i, j], iy[i, j]]) == 0:
                        u[i, j] = 0
                        v[i, j] = 0
                    else:
                        s = -it[i, j] / np.linalg.norm([ix[i, j], iy[i, j]])
                        n = [ix[i, j], iy[i, j]] / np.linalg.norm([ix[i, j], iy[i, j]])
                        uv = n * s
                        u[i, j] = uv[0]
                        v[i, j] = uv[1]

                else:
                    u[i, j] = 0
                    v[i, j] = 0

        # Calculating magnitude ||grad(I)|| = sqrt(u^2 + v^2)
        optic_flow = np.sqrt(u ** 2 + v ** 2)
        # Show and save
        cv2.imshow('thershold - {}'.format(t), optic_flow)
        cv2.imwrite('optical_flow\\threshold - {}.png'.format(t), optic_flow)

    cv2.waitKey(0)


if __name__ == '__main__':
    img1 = cv2.imread('data\\frame1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('data\\frame2.png', cv2.IMREAD_GRAYSCALE)

    # 2 blured frames
    img1 = cv2.GaussianBlur(img1, (5, 5), 1.5)
    img2 = cv2.GaussianBlur(img2, (5, 5), 1.5)

    # Getting gradients Ix, Iy, It
    ix, iy, it = get_grad(img1, img2)
    # Getting optical flow
    optical_flow(ix, iy, it)
