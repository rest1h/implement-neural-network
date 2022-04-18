import logging

import numpy as np

from activation import sigmoid
from loss import mean_squared_error
from optimize import Optimizer

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class FullyConnectedNeuralNetwork:
    def __init__(
        self,
        in_dim_n: int,
        out_dim_n: int,
        learning_rate: float,
        batch_size: int,
        layer_type: str,
        optim_fn: str = "sgd",
        clip_grad_l2: float = None,
    ):
        self.in_dim_n = in_dim_n
        self.out_dim_n = out_dim_n
        self.weight = np.random.rand(in_dim_n, out_dim_n)
        self.x_in = None
        self.layer_type = layer_type or None
        self.EI = np.zeros((batch_size, out_dim_n))
        self.optim = Optimizer(optim_fn, learning_rate, clip_grad_l2)

    def forward(self, x: np.array):
        self.x_in = np.copy(x)
        logger.debug(f"{self.x_in.shape = } {self.weight.shape = }")
        x_out = self.x_in @ self.weight
        logger.debug(f"{x_out.shape = }")
        logger.debug(f"{self.weight = }")

        return x_out

    def backward(
        self, x_out: np.array, y_true: np.array = None, agg_EI: np.array = None
    ):
        if self.layer_type == "out":
            logger.debug(f"{x_out.shape =}, {y_true.shape = }")
            EI = (x_out - y_true) @ x_out.T @ (1.0 - x_out)
            grad = self.x_in.T @ EI
            self.weight -= self.optim.optimize(grad)
            logger.debug(f"{EI.shape =}, {self.weight.shape = }")

            return np.einsum("ik,kj->ij", EI, self.weight.T)

        elif not self.layer_type == "hidden":
            logger.debug(f"{self.EI.shape =} {x_out.shape =}")
            self.EI += agg_EI @ (x_out.T @ (1.0 - x_out))
            logger.debug(f"{self.EI.shape =} {self.x_in.shape =}")
            grad = self.x_in.T @ self.EI
            self.weight -= self.optim.optimize(grad)


class BasicNeuralNetwork:
    def __init__(
        self,
        in_dim_n: int,
        hidden_dim_n: int,
        out_dim_n: int,
        epoch: int,
        learning_rate: float,
        batch_size: int,
        optimize_fn: str = "sgd",
        clip_grad_l2: float = None,
    ):
        self.hidden_layer = FullyConnectedNeuralNetwork(
            in_dim_n,
            hidden_dim_n,
            learning_rate,
            batch_size,
            "hidden",
            optimize_fn,
            clip_grad_l2,
        )
        self.out_layer = FullyConnectedNeuralNetwork(
            hidden_dim_n,
            out_dim_n,
            learning_rate,
            batch_size,
            "out",
            optimize_fn,
            clip_grad_l2,
        )
        self.epoch = epoch
        self.batch_size = batch_size

    def train(self, x_in: np.array, y: np.array):
        x_norm = self._normalize(x_in)
        results = []
        max_epoch = self.epoch
        while self.epoch > 0:
            x_iter = self._batch(x_norm)
            y_iter = self._batch(y)
            for n_iter, (x, y_true) in enumerate(zip(x_iter, y_iter)):
                result = []
                x1 = self.hidden_layer.forward(x)
                y1 = sigmoid(x1)
                x2 = self.out_layer.forward(y1)
                y2 = sigmoid(x2)

                loss = mean_squared_error(y2, y_true)
                print(
                    f"Epoch: {max_epoch - self.epoch}, "
                    f"Iteration: {n_iter}, Loss: {loss}"
                )
                result.append(loss)
                results.append(sum(result))

                agg_EI = self.out_layer.backward(y2, y_true=y_true)
                self.hidden_layer.backward(y1, y_true=y_true, agg_EI=agg_EI)
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
