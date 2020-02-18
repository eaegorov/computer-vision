import numpy as np
from matplotlib import pyplot as plt
import torchvision
import pickle
import pandas as pd
import seaborn as sn

K = 10  # Количество классов

# Функция, которая возвращает предсказания
def predict(X, W, B):
    y = softmax(X @ W + B)
    return y


# Функция подсчета accuracy
def accuracy(predicted, real):
    correct = 0
    total = real.shape[0]
    for i in range(total):
        p = predicted[i, :]
        t = real[i, :]
        if np.argmax(p) == np.argmax(t):
            correct += 1

    return round((correct / total) * 100, 2)


# Softmax
def softmax(Z):
    for i in range(len(Z)):
        a = np.max(Z[i, :])
        Z[i, :] -= a
        Z[i, :] = np.exp(Z[i, :]) / np.sum(np.exp(Z[i, :]))

    return Z


# Вычисление градиента
def E_gradient(X, W, B, T):
    lambd = 0.0005  # Regularization coefficient
    Y = predict(X, W, B)
    w_grad = (Y - T).T @ X

    U = np.ones((X.shape[0], 1))
    b_grad = (Y - T).T @ U

    return b_grad.T, w_grad.T + lambd * W


# Мини-батчевый градиентный спуск
def gradient_descent(x_train, y_train, x_test, y_test, lr):
    # Initialization
    w_next, b_next = initialization(x_train, K)

    best_acc = 0
    best_it = -1
    best_w = 0
    best_b = 0
    eps = 0.001
    batch_size = 32
    optimize = True
    iteration = 1
    while optimize and iteration <= 1000000:
        # Mini-batch generation
        idxs = np.random.randint(0, len(x_train), size=batch_size)
        batch = np.array([x_train[i, :] for i in idxs]).reshape(batch_size, x_train.shape[1])
        t = np.array([y_train[i, :] for i in idxs]).reshape(batch_size, y_train.shape[1])

        w_old = w_next
        b_old = b_next
        b_grad, w_grad = E_gradient(batch, w_old, b_old, t)

        w_next = w_old - lr * w_grad
        b_next = b_old - lr * b_grad

        if iteration % 100000 == 0:
            train_predictions = predict(x_train, w_next, b_next)
            test_predictions = predict(x_test, w_next, b_next)
            train_acc = accuracy(train_predictions, y_train)
            test_acc = accuracy(test_predictions, y_test)
            print('Iteration: {}'.format(iteration))
            print('Train accuracy:', train_acc)
            print('Test accuracy:', test_acc)
            print('-------------------')
            if test_acc > best_acc:
                best_acc = test_acc
                best_it = iteration
                best_w, best_b = w_next, b_next

        norm = np.sqrt(np.sum((w_next - w_old) ** 2))
        if norm < eps:
            break
        iteration += 1

    print('The best accuracy on {} iteration'.format(best_it))
    # Saving params
    with open('weights.pkl'.format(iteration), 'wb') as f:
        pickle.dump(best_w, f)
    with open('bias.pkl'.format(iteration), 'wb') as f:
        pickle.dump(best_b, f)

    return best_w, best_b

# Weights and bias initialization
def initialization(x_train, num_classes):
    var = 1 / (len(x_train) ** 0.5)
    W = np.random.normal(0.0, var, (x_train.shape[1], num_classes))
    B = np.random.normal(0.0, var, (num_classes, 1))
    return W, B.T


# Converting labels to one-hot-encoding vector
def one_hot_encoding(x, y, num_classes):
    y = np.array(y)
    y_one_hot = np.zeros((x.shape[0], num_classes))
    for i in range(len(y)):
        y_one_hot[i, y[i]] = 1

    return y_one_hot


def confusionMaxtrix(predicted, true, num_classes):
    size = (num_classes, num_classes)
    conf_matrix = np.zeros(size, dtype=int)
    total = true.shape[0]

    for i in range(total):
        p = np.argmax(predicted[i, :])
        t = np.argmax(true[i, :])
        conf_matrix[t, p] += 1

    # Visualization confusion matrix
    # df_cm = pd.DataFrame(conf_matrix, index=[i for i in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']],
    #                            columns=[i for i in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']])
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True, fmt='d')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    return conf_matrix


# Normalization
def featureNormalization(x):
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    x = (x - mean) / std

    return x


def pictures(x_test, y_test, predictions):
    idx_bad = []
    pred_bad = []
    idx_best = []
    pred_best = []

    for i in range(len(predictions)):
        p = np.argmax(predictions[i, :])
        t = np.argmax(y_test[i, :])
        if p != t:
            idx_bad.append(i)
            pred_bad.append(np.max(predictions[i, :]))
        else:
            idx_best.append(i)
            pred_best.append(np.max(predictions[i, :]))

    pred_bad = np.array(pred_bad)
    pred_best = np.array(pred_best)
    pred_bad_sort = pred_bad.argsort()
    pred_best_sort = pred_best.argsort()

    # Find top mistakes
    bad = []
    for i in range(len(pred_bad_sort) - 1, len(pred_bad_sort) - 4, -1):
        idx = idx_bad[pred_bad_sort[i]]
        bad.append(idx)

    # Find top best predictions
    best = []
    for i in range(len(pred_best_sort) - 1, len(pred_best_sort) - 4, -1):
        idx = idx_best[pred_best_sort[i]]
        best.append(idx)

    # Images
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10, 5))
    plt.title('Top mistakes')
    for i in range(len(bad)):
        plt.subplot(1, len(bad), i + 1)
        img = x_test[bad[i], :].reshape(32, 32, 3)
        plt.title(labels[np.argmax(y_test[bad[i], :])] + '/' + labels[np.argmax(predictions[bad[i], :])])
        plt.imshow(img)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Best predictions')
    for i in range(len(best)):
        plt.subplot(1, len(best), i + 1)
        img = x_test[best[i], :].reshape(32, 32, 3)
        plt.title(labels[np.argmax(y_test[best[i], :])])
        plt.imshow(img)
    plt.show()


# Train mode
def train(x_train, y_train, x_test, y_test, learning_rate):
    w, b = gradient_descent(x_train, y_train, x_test, y_test, learning_rate)
    print('Train is finished!')

    return w, b


# Eval mode
def eval(x_train, y_train, x_test, y_test):
    weights_name = 'weights.pkl'
    with open(weights_name, 'rb') as f:
        w = pickle.load(f)

    bias_name = 'bias.pkl'
    with open(bias_name, 'rb') as f:
        b = pickle.load(f)

    print('Evaluation:')
    train_predictions = predict(x_train, w, b)
    print('Train accuracy:', accuracy(train_predictions, y_train))
    test_predictions = predict(x_test, w, b)
    print('Test accuracy:', accuracy(test_predictions, y_test))

    return test_predictions


if __name__ == '__main__':
    # Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False)

    # Train data
    x_train = trainset.train_data
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = trainset.train_labels

    # Test data
    x_test = testset.test_data
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = testset.test_labels

    # Converting to one-hot encoding vector
    y_train = one_hot_encoding(x_train, y_train, K)
    y_test = one_hot_encoding(x_test, y_test, K)

    # Standartization
    x_train_norm = featureNormalization(x_train)
    x_test_norm = featureNormalization(x_test)

    # Train
    learning_rate = 0.001
    # w, b = train(x_train_norm, y_train, x_test_norm, y_test, learning_rate=learning_rate)

    # Evaluation
    test_pred = eval(x_train_norm, y_train, x_test_norm, y_test)

    # Confusion Matrix for the TEST SET
    CM = confusionMaxtrix(test_pred, y_test, K)
    print('Confusion matrix:', CM)

    # Top-3 mistake and best predictions on images
    # pictures(x_test, y_test, test_pred)
