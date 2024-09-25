import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# softmax

def get_softmax(arr):
    # arr = np.random.randn(10, 4)
    exp_arr = np.exp(arr)
    answer = exp_arr / (np.sum(exp_arr, axis=1, keepdims=True))
    return answer


# feed_forward
def forward(x, w1, b1, w2, b2):
    z = np.tanh(x.dot(w1) + b1)
    k = np.tanh(z.dot(w2) + b2)
    final = get_softmax(k)
    return final


'''
x1 = np.random.randn(20, 2) + np.array([0, -2])
x2 = np.random.rand(20, 2) + np.array([2, -2])
x3 = np.random.rand(20, 2) + np.array([-2, 2])
Y = np.array([0]*20 + [1]*20 + [2]*20)
X = np.vstack([x1, x2, x3])
print(X)
print(X[:, 0])
print(X[:, 1])
print(Y)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
plt.show()
'''


# process

def get_data():
    df = pd.read_csv("ecommerce_data.csv")
    data = df.to_numpy()

    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, :(D - 1)] = X[:, :(D - 1)]
    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1
    X = X2
    x_train = X[:-100]
    y_train = Y[:-100]
    x_test = X[-100:]
    y_test = Y[-100:]
    print(x_train[1])
    print(x_train[4])
    for i in (1, 2):
        m = np.mean(x_train[:, i])
        print(m)
        s = np.std(x_train[:, i])
        print(s)
        x_train[:, i] = (x_train[:, i] - m) / s
        x_test[:, i] = (x_test[:, i] - m) / s

    print(x_train[1])
    print(x_train[4])
    return x_train, y_train, x_test, y_test


'''
x, y, _, _ = get_data()
d = x.shape[1]
m = 5
k = len(set(y))
w1 = np.random.rand(d, m)
b1 = np.zeros(m)
w2 = np.random.rand(m, k)
b2 = np.zeros(k)
p_y_given_x = forward(x, w1, b1, w2, b2)
print(p_y_given_x.shape)
'''
x = np.array([[1, 2]])
w1 = np.array([[1, 1], [1, 0]])
w2 = np.array([[0, 1], [1, 1]])
z = np.tanh(x.dot(w1.transpose()))
print(get_softmax(z.dot(w2.transpose())))