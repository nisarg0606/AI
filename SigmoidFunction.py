import numpy as np

X = [0.5, 2.5]
Y = [0.2, 0.9]


def f(w, b, x):  # sigmoid function / f(x)
    return 1.0 / (1.0 + np.exp(-(w*x + b)))


def error(w, b):  # error function / L(w,b)
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y)**2
    return err


def grad_b(w, b, x, y):  # gradient of b
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)


def grad_w(w, b, x, y):  # gradient of w
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


def do_gradient_descent():
    w, b, eta, max_epochs = -2, -2, 1.0, 100
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w = w - eta * dw
        b = b - eta * db
        err = error(w, b)
    print("weights --> ", w, "Bias --> ", b)
    print("Error --> ", err)
    # return w, b


do_gradient_descent()
# w, b = do_gradient_descent()
# print("w = ", w)
# print("b = ", b)

# print("f(0.5, w, b) = ", f(0.5, w, b))
