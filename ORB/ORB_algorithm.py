import cv2
import numpy as np
import pickle
from scipy.ndimage.filters import maximum_filter


# FAST keypoing detector
def FAST(img):
    t = 10

    interesting_points = []
    for i in range(15, img.shape[0] - 15):
        for j in range(15, img.shape[1] - 15):
            Ip = img[i, j]
            Ic = [img[i - 3, j], img[i, j + 3], img[i + 3, j], img[i, j - 3]]
            check1 = [Ip > Ic[0] + t, Ip > Ic[1] + t, Ip > Ic[2] + t, Ip > Ic[3] + t]
            check2 = [Ip < Ic[0] - t, Ip < Ic[1] - t, Ip < Ic[2] - t, Ip < Ic[3] - t]
            if check1.count(True) >= 3 or check2.count(True) >= 3:
                interesting_points.append((i, j))

    return interesting_points


# Gradient
def get_grad(img):
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    kernel = np.array([-1, 8, 0, -8, 1]) / 12
    kernel = kernel.reshape(1, 5)
    Ix = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel.T, borderType=cv2.BORDER_REFLECT)

    return Ix, Iy


# Filtering FAST points with Harris criterion
def harris_corner_filter(img, points):
    # Gradients of image
    Ix, Iy = get_grad(img)

    # Gaussian mask
    mask = np.array([1, 4, 6, 4, 1]) / 16
    mask = mask.reshape(1, 5)
    # Gaussian kernel
    w = mask.T @ mask

    # Getting Ix^2, Iy^2, IxIy
    ix_squared = Ix * Ix
    iy_squared = Iy * Iy
    IxIy = Ix * Iy

    # Components a, b and c for matrix M
    a = cv2.filter2D(ix_squared, ddepth=cv2.CV_32F, kernel=w, borderType=cv2.BORDER_REFLECT)
    b = cv2.filter2D(IxIy, ddepth=cv2.CV_32F, kernel=w, borderType=cv2.BORDER_REFLECT)
    c = cv2.filter2D(iy_squared, ddepth=cv2.CV_32F, kernel=w, borderType=cv2.BORDER_REFLECT)

    # Filtering points
    harris_responses = []
    harris_points = []
    k = 0.04
    for x, y in points:
        M = np.array([[a[x, y], b[x, y]],
                      [b[x, y], c[x, y]]]).reshape(2, 2)

        R = np.linalg.det(M) - k * (np.trace(M) ** 2)
        if R > 0:
            harris_responses.append(R)
            harris_points.append((x, y))

    # Getting N important points
    N = 200
    harris_responses = np.array(harris_responses)
    indices = harris_responses.argsort()

    # Top N points
    count_points = len(harris_responses)
    indices = indices[count_points - N:]
    topN_harris_responses = [harris_responses[idx] for idx in indices]
    topN_harris_points = [harris_points[idx] for idx in indices]

    # Non-maximum supression, to remote bad points and a set of similar points in neighborhood
    top_points = NMS(img, topN_harris_responses, topN_harris_points)

    return top_points


# Non-maximum supression
def NMS(img, responses, points):
    responses_matrix = np.zeros(img.shape)

    # Fill the matrix
    for k, (i, j) in enumerate(points):
        responses_matrix[i, j] = responses[k]

    coef = 0.02
    threshold = np.max(responses_matrix) * coef
    neighbors_size = 5
    maximums = maximum_filter(responses_matrix, neighbors_size)
    maximums = (responses_matrix == maximums)

    peaks = []
    for i in range(maximums.shape[0]):
        for j in range(maximums.shape[1]):
            if (maximums[i, j] == True) and (responses_matrix[i, j] > threshold):
                peaks.append((i, j))

    return peaks


# Calculating moments
def get_moments(img, points):
    m10 = []
    m01 = []
    for x, y in points:
        s10 = 0
        s01 = 0
        for i in range(x - 15, x + 15):
            for j in range(y - 15, y + 15):
                s10 += i * img[i, j]
                s01 += j * img[i, j]

        m10.append(s10)
        m01.append(s01)

    # Calculating theta of each keypoint with m10 and m01
    thetas = []
    for i in range(len(m10)):
        t = np.arctan2(m01[i], m10[i])
        thetas.append(t)

    return thetas


# BRIEF descriptor
def BRIEF(img, keypoints, thetas):
    # Smooth image
    img = cv2.GaussianBlur(img, (9, 9), 1.5)

    # Getting pairs of points
    file = 'orb_descriptor_positions.txt'  # Points from file
    pair_points = np.loadtxt(file)

    # Building descriptors for each keypoint
    n = 256
    descriptors = np.empty((len(keypoints), n))
    for k, (x, y) in enumerate(keypoints):
        for i in range(pair_points.shape[0]):
            # Turn points for the angle of keypoint
            R_theta = np.array([[np.cos(thetas[k]), -np.sin(thetas[k])],
                                [np.sin(thetas[k]), np.cos(thetas[k])]]).reshape(2, 2)

            p1 = R_theta @ np.array([pair_points[i, 0], pair_points[i, 1]]).reshape(2, 1)
            p2 = R_theta @ np.array([pair_points[i, 2], pair_points[i, 3]]).reshape(2, 1)

            # Test
            point1_coord = [x + int(p1[0]), y + int(p1[1])]
            point2_coord = [x + int(p2[0]), y + int(p2[1])]
            if point1_coord[0] >= img.shape[0] or point1_coord[1] >= img.shape[1] or point2_coord[0] >= img.shape[0] or point2_coord[1] >= img.shape[1]:
                descriptors[k, i] = 0
            else:
                Ip1 = img[point1_coord[0], point1_coord[1]]
                Ip2 = img[point2_coord[0], point2_coord[1]]
                if Ip1 < Ip2:
                    descriptors[k, i] = 1
                else:
                    descriptors[k, i] = 0

    return descriptors


# Saving
def save(descriptors):
    with open('descriptors.pkl', 'wb') as f:
        pickle.dump(descriptors, f)


if __name__ == '__main__':
    # Loading image in greyscale
    img = cv2.imread('data\\blox.jpg', cv2.IMREAD_GRAYSCALE)

    # FAST detector points
    interest_points = FAST(img)

    # Filtering points with Harris cruterion and getting top-N
    top_points = harris_corner_filter(img, interest_points)

    # Getting angles of moments for each keypoint
    thetas = get_moments(img, top_points)

    # Getting descriptor for each keypoint
    descriptors = BRIEF(img, top_points, thetas)

    # Saving descriptors
    save(descriptors)

    # Showing points
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for y, x in top_points:
        cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=2)

    cv2.imshow('Detected keypoints', img)
    cv2.waitKey(0)
