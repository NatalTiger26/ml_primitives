import numpy as np

def f(x):
    return (x-3)**2

def grad_f(x):
    return 2*(x-3)

def gradient_descent_1d(start, lr, steps):
    pt = start
    history = [pt]
    for _ in range(steps):
        pt -= lr * grad_f(pt)
        history.append(pt)
    return history[-1], history

def f2(x, y):
    return x**2 + 5 * y**2

def grad_f2(x, y):
    return 2 * x,  10 * y

def gradient_descent_2d(start, lr, steps):
    x, y = start
    history = [(x, y)]
    for _ in range(steps):
        gx, gy = grad_f2(x, y)
        x -= lr * gx
        y -= lr * gy
        history.append((x, y))
    return history[-1], np.array(history)
