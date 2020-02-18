from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Gaussian kernel generation
def make_kernel(kernelSize, sigma):
    kernel = np.zeros((kernelSize, kernelSize))
    coef_for_center = (2 * 3 * sigma) / (kernelSize * 2)

    div = 0
    x = -3 * sigma + coef_for_center
    y = 3 * sigma - coef_for_center
    for i in range(kernelSize):
        x_start_line = x
        for j in range(kernelSize):
            kernel[i, j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            div += kernel[i, j]
            x += coef_for_center * 2
        x = x_start_line
        y -= coef_for_center * 2

    kernel /= div
    return kernel


# Add noise
def make_noise(img):
    noise = np.random.normal(loc=0.0, scale=10.0, size=(img.shape[0], img.shape[1], img.shape[2]))
    img_noise = img + noise
    img_noise = cv2.normalize(img_noise, img_noise, 0, 255, cv2.NORM_MINMAX, dtype=-1)  # Normalized image
    img_noise = img_noise.astype(np.uint8)

    return img_noise


# Blur
def bluring_imgage(img, kernel):
    stride = 1
    padding = int((kernel.shape[0] - 1) / 2)
    new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    kernel_size = kernel.shape[0]
    print(new_img)

    conv_img = np.empty((img.shape[0], img.shape[1]))

    for i in range(0, new_img.shape[0], stride):
        for j in range(0, new_img.shape[1], stride):
            conv_window = new_img[i: i + kernel_size, j: j + kernel_size]
            if conv_window.shape[0] < kernel_size or conv_window.shape[1]  < kernel_size:
                break
            conv_img[i, j] = (new_img[i: i + kernel_size, j: j + kernel_size] * kernel).sum()

    conv_img = cv2.normalize(conv_img, conv_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)  # Normalized image
    conv_img = conv_img.astype(np.uint8)

    return conv_img


if __name__ == '__main__':
    path = 'lena_color_256.tif'
    img = misc.imread(path)

    gaussian_kernel = make_kernel(kernelSize=9, sigma=100)  # Gaussian kernel

    img_noise = make_noise(img=img)  # Image with noise

    img_gray = cv2.cvtColor(img_noise, cv2.COLOR_RGB2GRAY)  # Converting to gray image

    plt.hist(img_gray) # Histogram of gray image
    plt.show()

    img_blur = bluring_imgage(img=img_gray, kernel=gaussian_kernel) # Blured image
    cv2.imshow('a', img_blur)
    cv2.waitKey()