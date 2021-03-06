import numpy as np


class Optimizer:
    def __init__(self, optim_fn: str, lr: float, clip_grad_l2: float, momentum: float = None):
        self.optim_fn = optim_fn
        self.lr = lr
        self.clip_grad_l2 = clip_grad_l2
        self.square_diag_grad = None
        self.use_momentum = momentum
        self.momentum = momentum
        self.grad_shift = 0

    def optimize(self, grad: np.array):
        if self.optim_fn.lower() == "sgd":
            return self.sgd(grad)
        elif self.optim_fn.lower() == "adagrad":
            return self.adagrad(grad)

    def sgd(self, grad_in: np.array):
        grad = np.copy(grad_in)
        if self.clip_grad_l2:
            grad = self._clip_grad(grad)
        if self.use_momentum:
            self.grad_shift =  self.lr * grad + self.momentum * self.grad_shift
            return self.grad_shift
        return self.lr * grad

    def adagrad(self, grad_in: np.array):
        grad = np.copy(grad_in)
        if self.clip_grad_l2:
            grad = self._clip_grad(grad)
        if self.square_diag_grad is None:
            self.square_diag_grad = np.zeros(grad.shape)
        self.square_diag_grad += np.diag(grad) @ np.diag(grad).T
        return self.lr * np.diag(grad) / np.sqrt(self.square_diag_grad + 1e-8)

    def _clip_grad(self, grad_in: np.array):
        grad = np.copy(grad_in)
        if any(abs(grad) > self.clip_grad_l2):
            return grad / np.linalg.norm(grad)
        else:
            return grad
