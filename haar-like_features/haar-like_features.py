import cv2
import numpy as np
import glob
import pickle


# Стандартизация изображений
def standartization(img):
    img = np.array(img, dtype=np.float32)
    return (img - np.mean(img)) / np.std(img)


# Получение интегрального изображения
def get_integral_image(img):
    ii = np.empty((img.shape))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0:
                ii[i, j] = np.sum(img[0, :j + 1])
            else:
                ii[i, j] = img[i, j] + np.sum(img[i, :j]) + ii[i - 1, j]

    return ii


# Два прямоугольника, разделенных вертикально
def x2_feature(integral_image):
    img_size = integral_image.shape[0]
    features = []
    for width in range(1, img_size // 2):
        for height in range(1, img_size):
            h, w = height, width * 2  # Размер паттерна
            for x in range(img_size - h):
                for y in range(img_size - w):
                    white = integral_image[x + h, y + int(w / 2)] - integral_image[x, y + int(w / 2)] - integral_image[x + h, y] + integral_image[x, y]
                    dark = integral_image[x + h, y + w] - integral_image[x, y + w] - integral_image[x + h, y + int(w / 2)] + integral_image[x, y + int(w / 2)]
                    haar_feature = white - dark
                    features.append(haar_feature)

    return np.array(features)


# Два прямоугольника, разделенных горизонтально
def y2_feature(integral_image):
    img_size = integral_image.shape[0]
    features = []
    for width in range(1, img_size):
        for height in range(1, img_size // 2):
            h, w = height * 2, width  # Размер паттерна
            for x in range(img_size - h):
                for y in range(img_size - w):
                    white = integral_image[x + int(h / 2), y + w] - integral_image[x, y + w] - integral_image[x + int(h / 2), y] + integral_image[x, y]
                    dark = integral_image[x + h, y + w] - integral_image[x + int(h / 2), y + w] - integral_image[x + h, y] + integral_image[x + int(h / 2), y]
                    haar_feature = white - dark
                    features.append(haar_feature)

    return np.array(features)


# Три прямоугольника, разделенных вертикально
def x3_feature(integral_image):
    img_size = integral_image.shape[0]
    features = []
    for width in range(1, img_size // 3):
        for height in range(1, img_size):
            h, w = height, width * 3  # Размер паттерна
            for x in range(img_size - h):
                for y in range(img_size - w):
                    white = (integral_image[x + h, y + int(w / 3)] - integral_image[x, y + int(w / 3)] - integral_image[x + h, y] + integral_image[x, y]) + \
                            (integral_image[x + h, y + w] - integral_image[x, y + w] - integral_image[x + h, y + int(w / 3) * 2] + integral_image[x, y + int(w / 3) * 2])
                    dark = integral_image[x + h, y + int(w / 3) * 2] - integral_image[x, y + int(w / 3) * 2] - integral_image[x + h, y + int(w / 3)] + integral_image[x, y + int(w / 3)]
                    haar_feature = white - dark
                    features.append(haar_feature)

    return np.array(features)


# Три прямоугольника, разделенных горизонтально
def y3_feature(integral_image):
    img_size = integral_image.shape[0]
    features = []
    for width in range(1, img_size):
        for height in range(1, img_size // 3):
            h, w = height * 3, width  # Размер паттерна
            for x in range(img_size - h):
                for y in range(img_size - w):
                    white = (integral_image[x + int(h / 3), y + w] - integral_image[x, y + w] - integral_image[x + int(h / 3), y] + integral_image[x, y]) + \
                            (integral_image[x + h, y + w] - integral_image[x + int(h / 3) * 2, y + w] - integral_image[x + h, y] + integral_image[x + int(h / 3) * 2, y])
                    dark = integral_image[x + int(h / 3) * 2, y + w] - integral_image[x + int(h / 3), y + w] - integral_image[x + int(h / 3) * 2, y] + integral_image[x + int(h / 3), y]
                    haar_feature = white - dark
                    features.append(haar_feature)

    return np.array(features)


# Четрыре прямоугольника, разделенных вертикально и горизонтально
def x4y4_feature(integral_image):
    img_size = integral_image.shape[0]
    features = []
    for width in range(1, img_size // 2):
        for height in range(1, img_size // 2):
            h, w = height * 2, width * 2  # Размер паттерна
            for x in range(img_size - h):
                for y in range(img_size - w):
                    white = (integral_image[x + int(h / 2), y + int(w / 2)] - integral_image[x, y + int(w / 2)] - integral_image[x + int(h / 2), y] + integral_image[x, y]) + \
                            (integral_image[x + h, y + w] - integral_image[x + int(h / 2), y + w] - integral_image[x + h, y + int(w / 2)] + integral_image[x + int(h / 2), y + int(w / 2)])
                    dark = (integral_image[x + int(h / 2), y + w] - integral_image[x, y + w] - integral_image[x + int(h / 2), y + int(w / 2)] + integral_image[x, y + int(w / 2)]) + \
                           (integral_image[x + h, y + int(w / 2)] - integral_image[x + int(h / 2), y + int(w / 2)] - integral_image[x + h, y] + integral_image[x + int(h / 2), y])
                    haar_feature = white - dark
                    features.append(haar_feature)

    return np.array(features)


# Вычисление haar-like features
def haar_features_extracting(integral_image):
    first_feature = x2_feature(integral_image)
    second_feature = y2_feature(integral_image)
    third_feature = x3_feature(integral_image)
    fourth_feature = y3_feature(integral_image)
    fifth_feature = x4y4_feature(integral_image)

    features = np.concatenate((first_feature, second_feature, third_feature, fourth_feature, fifth_feature))

    return features


# Сбор всех признаков в датасет
def generate_dataset(faces, nonfaces):
    x_faces = []
    y_faces = [1 for i in range(len(faces))]
    print('Face images processing...')
    for filename in faces:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = standartization(img)
        integral_image = get_integral_image(img)
        features = haar_features_extracting(integral_image)
        x_faces.append(features)

    x_faces = np.array(x_faces).reshape(len(faces), len(x_faces[0]))
    y_faces = np.array(y_faces)

    x_nonfaces = []
    y_nonfaces = [0 for i in range(len(nonfaces))]
    print('Non-face images processing...')
    for filename in nonfaces:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = standartization(img)
        integral_image = get_integral_image(img)
        features = haar_features_extracting(integral_image)
        x_nonfaces.append(features)

    x_nonfaces = np.array(x_nonfaces).reshape(len(nonfaces), len(x_nonfaces[0]))

    x_train = np.vstack((x_faces, x_nonfaces))
    y_train = np.hstack((y_faces, y_nonfaces))
    print('Data are ready!')

    return x_train, y_train


# Сохранение данных по частям ввиду нехватки памяти
def separate_dataset(faces_train, nonfaces_train):
    k1 = int(len(faces_train) // 4)
    k2 = int(len(nonfaces_train) // 4)

    x_train1, y_train1 = generate_dataset(faces_train[:k1], nonfaces_train[:k1])
    pickle_data(x_train1, y_train1, 1)

    x_train2, y_train2 = generate_dataset(faces_train[k1:2 * k1], nonfaces_train[k2:2 * k2])
    pickle_data(x_train2, y_train2, 2)

    x_train3, y_train3 = generate_dataset(faces_train[2 * k1:3 * k1], nonfaces_train[2 * k2:3 * k2])
    pickle_data(x_train3, y_train3, 3)

    x_train4, y_train4 = generate_dataset(faces_train[3 * k1:], nonfaces_train[3 * k2:])
    pickle_data(x_train4, y_train4, 4)


# Сериализация данных
def pickle_data(x, y, n):
    with open('x_train{}.pkl'.format(n), 'wb') as f:
        pickle.dump(x, f)
    with open('y_train{}.pkl'.format(n), 'wb') as f:
        pickle.dump(y, f)


def collect_data(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4):
    with open(x_1, 'rb') as f:
        x_train1 = pickle.load(f)
    with open(y_1, 'rb') as f:
        y_train1 = pickle.load(f)

    with open(x_2, 'rb') as f:
        x_train2 = pickle.load(f)
    with open(y_2, 'rb') as f:
        y_train2 = pickle.load(f)

    with open(x_3, 'rb') as f:
        x_train3 = pickle.load(f)
    with open(y_3, 'rb') as f:
        y_train3 = pickle.load(f)

    with open(x_4, 'rb') as f:
        x_train4 = pickle.load(f)
    with open(y_4, 'rb') as f:
        y_train4 = pickle.load(f)

    x_train = np.vstack((x_train1, x_train2))
    y_train = np.hstack((y_train1, x_train2))
    x_train = np.vstack((x_train, x_train3))
    y_train = np.hstack((y_train, y_train3))
    x_train = np.vstack((x_train, x_train4))
    y_train = np.hstack((y_train, y_train4))

    return x_train, y_train



if __name__ == '__main__':
    faces_train = glob.glob('train\\face\\*.pgm')
    nonfaces_train = glob.glob('train\\non-face\\*.pgm')
    print(len(faces_train))
    print(len(nonfaces_train))

    # separate_dataset(faces_train, nonfaces_train)
    x_train, y_train = collect_data('x_train1.pkl', 'y_train1.pkl', 'x_train2.pkl', 'y_train2.pkl', 'x_train3.pkl', 'y_train3.pkl', 'x_train4.pkl', 'y_train4.pkl')
    print(x_train.shape)
    print(y_train.shape)




