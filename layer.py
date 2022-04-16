import numpy as np
from activation import sigmoid
from loss import mean_squared_error

# TODO: no bias, batch size = 1 
class FullyConnectedNeuralNetwork():
    def __init__(self, in_dim_n: int, 
                 out_dim_n: int,
                 learning_rate: float,
                 activation_type: str=None):
        self.in_dim_n = in_dim_n
        self.out_dim_n = out_dim_n
        self.weight = np.random.rand(in_dim_n, out_dim_n)
        self.x_in = None
        self.x_out = None
        self.learning_rate = learning_rate
        self.activation_type = activation_type 
        self.EI = np.zeros((1000, 8))
           
    def forward(self, x):
        self.x_in = x.copy()
        self.x_out = self.x_in @ self.weight
        return self.x_out
    
    def backward(self, y_true=None, EIP=None):
        if self.activation_type == "sigmoid":
            print(self.x_out, y_true)
            EIP = (self.x_out - y_true) * self.x_out.T * (1. - self.x_out)
            # EIP = (self.x_out - y_true) @ self.x_out.T @ (1. - self.x_out)
            self.weight -= self.learning_rate * self.x_in.T * EIP
            # self.weight -= self.learning_rate * self.x_in.T @ EIP
            return sum(EIP * self.weight.T)
            # return sum(EIP @ self.weight.T)
        elif self.activation_type == None:
            self.EI += EIP * (self.x_out.T * (1. - self.x_out))
            # self.EI += EIP @ (self.x_out.T @ (1. - self.x_out))
            self.weight -= self.learning_rate * self.x_in.T * self.EI
            # self.weight -= self.learning_rate * self.x_in.T @ self.EI
    
    def normalize(self, x):
        return (x - np.min(x))/(np.max(x) - np.min(x)) 
    
    
class BasicNeuralNetwork():
    def __init__(self, 
                 in_dim_n: int=2, 
                 hidden_dim_n: int=8, 
                 out_dim_n: int=1,
                 epoch: int=10,
                 learning_rate: float=0.01):
        self.hidden_layer = FullyConnectedNeuralNetwork(in_dim_n, 
                                                        hidden_dim_n, 
                                                        learning_rate)
        self.out_layer = FullyConnectedNeuralNetwork(hidden_dim_n, 
                                                     out_dim_n, 
                                                     learning_rate, 
                                                     "sigmoid")
        self.epoch = epoch
        self.x = None
    
    def train(self, x_in, y_true):       
        self.x = self.normalize(x_in)
         
        results = []
        while (self.epoch > 0):
            for iter, (x, y) in enumerate(zip(self.x, y_true)):
                result = []
                x = self.normalize(x)
                x = self.hidden_layer.forward(x)
                x = self.out_layer.forward(x)
                x = sigmoid(x)
                
                loss = mean_squared_error(x, y)
                print(f'Epoch: {self.epoch}, Iteration: {iter}, Loss: {loss}')
                result.append(loss)            
                self.epoch -= 1
                
                EIP = self.out_layer.backward(y_true=y)
                self.hidden_layer.backward(EIP=EIP)
                results.append(sum(result))
            
        return results
    
    def predict(self):
        pass
    
    def normalize(self, x):
        return (x - np.min(x))/(np.max(x) - np.min(x)) 
    
    def drop_out(x):
        pass
    
    