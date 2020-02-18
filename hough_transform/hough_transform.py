import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.filters import maximum_filter


# Преобразование Хафа, принимает бинарное изображение с выделенными границами
def hough_transform_lines(img_edges):
    height, width = img_edges.shape
    max_ro = int(round(np.sqrt(height ** 2 + width ** 2)))
    theta_numbers = np.linspace(-np.pi / 2, np.pi / 2, 500)

    accumulator = np.zeros((max_ro, len(theta_numbers)))
    for x in range(height):
        for y in range(width):
            if img_edges[x, y] > 0:
                for t in range(len(theta_numbers)):
                    ro = int(round(x * np.cos(theta_numbers[t]) + y * np.sin(theta_numbers[t])))
                    if ro >= 0:
                        accumulator[ro, t] += 1

    blured_accumulator = cv2.GaussianBlur(accumulator, (5, 5), 1)  # Сглаживание аккумулятора
    lines = find_local_maximums(blured_accumulator, 0.7)

    return theta_numbers, lines


# Поиск локальных максимумов в сглаженной матрице
def find_local_maximums(accumulator, coeff):
    peaks = []

    threshold = np.max(accumulator) * coeff
    neighbors_size = 5
    maximums = maximum_filter(accumulator, neighbors_size)
    maximums = (accumulator == maximums)
    for i in range(maximums.shape[0]):
        for j in range(maximums.shape[1]):
            if (maximums[i, j] == True) and (accumulator[i, j] > threshold):
                peaks.append((i, j))

    return peaks


# Отрисовка линий на исходном изображении
def draw_lines(image, thetas, lines):
    theta_numbers = thetas

    for r, t in lines:
        ro = r
        theta = theta_numbers[t]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = ro * a
        y0 = ro * b
        point1 = (int(x0 + 1500 * (-b)), int(y0 + 1500 * a))
        point2 = (int(x0 - 1500 * (-b)), int(y0 - 1500 * a))
        cv2.line(image, point1, point2, (0, 0, 255), 2, cv2.LINE_AA)


def hough_transform_circles(img_edges):
    height, width = img_edges.shape
    # max_r = 50 #min(height - height // 2, width - width // 2)
    r = 20
    theta_numbers = np.linspace(0, 2 * np.pi, 360)

    # accumulator = np.zeros((height, width, max_r))
    accumulator = np.zeros((height, width))

    for x in range(height):
        for y in range(width):
            #for r in range(max_r):
            for t in range(len(theta_numbers)):
                a = int(round(x - r * np.cos(theta_numbers[t])))
                b = int(round(y - r * np.sin(theta_numbers[t])))
                if a >= 0 and b >= 0 and a < height and b < width:
                    # accumulator[a, b, r] += 1
                    accumulator[a, b] += 1

    circles = find_local_maximums(accumulator, 0.9)

    return circles


def draw_circles(image, circles):
    for x, y in circles:
        center = (x, y)
        cv2.circle(image, center, 1, (0, 100, 100), 3)
        cv2.circle(image, center, 20, (255, 0, 255), 2)



if __name__ == '__main__':
    # Lines
    img = cv2.imread('Lines\\lines5.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)

    thetas, lines = hough_transform_lines(img_edges)

    cv2.imshow("Original image", img) # Исходное ихображение
    cv2.imshow("Edges", img_edges) # Границы
    draw_lines(img, thetas, lines)
    cv2.imshow("Detected Lines", img) # Линии
    cv2.waitKey(0)


    # Circles
    img = cv2.imread('Coins\\coins_75.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)

    circles = hough_transform_circles(img_edges)
    draw_circles(img, circles)
    cv2.imshow("Detected Circles", img)
    cv2.waitKey(0)

