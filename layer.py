import logging

import numpy as np

from activation import sigmoid
from loss import mean_squared_error

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class FullyConnectedNeuralNetwork:
    def __init__(
        self,
        in_dim_n: int,
        out_dim_n: int,
        learning_rate: float,
        batch_size: int,
        layer_type: str = None,
    ):
        self.in_dim_n = in_dim_n
        self.out_dim_n = out_dim_n
        self.weight = np.random.rand(in_dim_n, out_dim_n)
        self.x_in = None
        self.learning_rate = learning_rate
        self.layer_type = layer_type
        self.EI = np.zeros((batch_size, out_dim_n))
        self.threshold = 1
        self.grad = None

    def forward(self, x: np.array):
        self.x_in = np.copy(x)
        logger.debug(f"{self.x_in.shape = } {self.weight.shape = }")
        x_out = self.x_in @ self.weight
        logger.debug(f"{x_out.shape = }")
        logger.debug(f'{self.weight = }')

        return x_out

    def backward(self,
                 x_out: np.array,
                 y_true: np.array = None,
                 einsum_EIP: np.array = None):
        if self.layer_type == "out":
            logger.debug(f"{x_out.shape =}, {y_true.shape = }")
            EIP = (x_out - y_true) @ x_out.T @ (1.0 - x_out)
            self.grad = self.x_in.T @ EIP
            self._clip_grad()
            self.weight -= self.learning_rate * self.grad
            logger.debug(f"{EIP.shape =}, {self.weight.shape = }")
            return np.einsum("ik,kj->ij", EIP, self.weight.T)
        elif not self.layer_type == "hidden":
            logger.debug(f"{self.EI.shape =} {x_out.shape =}")
            self.EI += einsum_EIP @ (x_out.T @ (1.0 - x_out))
            logger.debug(f"{self.EI.shape =} {self.x_in.shape =}")
            self.grad = self.x_in.T @ self.EI
            self._clip_grad()
            self.weight -= self.learning_rate * self.grad

    def _clip_grad(self):
        if any(abs(self.grad) > self.threshold):
            self.grad /= np.linalg.norm(self.grad)


class BasicNeuralNetwork:
    def __init__(
        self,
        in_dim_n: int,
        hidden_dim_n: int,
        out_dim_n: int,
        epoch: int,
        learning_rate: float,
        batch_size: int,
    ):
        self.hidden_layer = FullyConnectedNeuralNetwork(
            in_dim_n, hidden_dim_n, learning_rate, batch_size, "hidden"
        )
        self.out_layer = FullyConnectedNeuralNetwork(
            hidden_dim_n, out_dim_n, learning_rate, batch_size, "out"
        )
        self.epoch = epoch
        self.batch_size = batch_size

    def train(self, x_in: np.array, y_true: np.array):
        x_norm = self._normalize(x_in)
        results = []
        max_epoch = self.epoch
        while self.epoch > 0:
            x_iter = self._batch(x_norm)
            y_iter = self._batch(y_true)
            for n_iter, (x, y) in enumerate(zip(x_iter, y_iter)):
                result = []
                x = self.hidden_layer.forward(x)
                x_out_1 = sigmoid(x)
                x = self.out_layer.forward(x_out_1)
                x_out_2 = sigmoid(x)

                loss = mean_squared_error(x_out_2, y)
                print(
                    f"Epoch: {max_epoch - self.epoch}, "
                    f"Iteration: {n_iter}, Loss: {loss}"
                )
                result.append(loss)

                einsum_EIP = self.out_layer.backward(x_out_2, y_true=y)
                self.hidden_layer.backward(x_out_1,
                                           y_true=y,
                                           einsum_EIP=einsum_EIP)
                results.append(sum(result))
            self.epoch -= 1

        return results

    def _batch(self, x_in: np.array):
        x = np.copy(x_in)
        end = len(x)
        for idx in range(0, end, self.batch_size):
            yield x[idx: min(idx + self.batch_size, end)]

    def predict(self):
        pass

    @staticmethod
    def _normalize(x):
        return x / np.linalg.norm(x)

    def drop_out(self, x):
        pass
