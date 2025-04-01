import numpy as np 
class Network():
    def __init__(self,sizes,layers):
        self.sizes = sizes
        self.layers = layers
        self.weights = [np.random.randn(x, y) * np.sqrt(1 / y) for x,y in zip(sizes[1:],sizes[:-1])]
        self.bias = [np.random.randn(x,1) * 0.1 for x in sizes[1:]]
    def forward_feed(self, x): 
        a = x
        for w,b,l in zip(self.weights, self.bias,self.layers):
            z = np.dot(w,a) + b
            a = self.activation(z,l)
        return a
    def backprop(self, x, t,lr):
        #first forward feed 
        func = []
        zact = [x] 
        for w,b,l in zip(self.weights, self.bias,self.layers):
            a = np.dot(w,zact[-1]) + b
            z = self.activation(a,l)
            func.append(a)
            zact.append(z)
        y = zact[-1]
        #calculate cost 
        cost = self.categorical_cross_entropy(t,y)
        #differentiate loss
        loss = (y-t)
        dweights = [np.zeros_like(w) for w in self.weights]
        dbias = [np.zeros_like(b) for b in self.bias] 
        dweights[-1] = np.dot(loss, zact[-2].transpose())
        dbias[-1] = loss
        for i in range(2,len(self.sizes)):
            loss = np.dot(self.weights[-i+1].T,loss) * self.relu_derivative(func[-i])
            dweights[-i] = np.dot(loss, zact[-i - 1].transpose())
            dbias[-i] = loss
        #apply change to weights
        for k in range(len(self.sizes)-1):
            self.weights[k] -= lr * dweights[k] 
            self.bias[k] -= lr * dbias[k]
        
        return cost

    def training(self, x_train, y_train ,lr,epoch,totalnum):
        j = 0
        for x,y in zip(x_train,y_train):
            his = []
            for i in range(epoch): 
                c = self.backprop(x,y,lr)
                if i %25 ==0:
                    print(f"{j} - {i} : cost {c:.4f}")
                    his.append(c)
            j+=1
            if j > totalnum:
                break
        return his
    def testing(self, x_test, y_test):
        correct = 0
        total = len(x_test)
        
        for x, y in zip(x_test, y_test):
            output = self.forward_feed(x)
            predicted_label = np.argmax(output, axis=0)  # Get index of max probability
            actual_label = np.argmax(y, axis=0)  # Get index of actual class
            
            if predicted_label == actual_label:
                correct += 1

        accuracy = (correct / total) * 100
        return accuracy
    def activation(self, x, l): 
        acti = {"r": self.relu, "sig": self.sigmoid, "sof": self.softmax}
        if l in acti:
            return acti[l](x)
        else:
            raise ValueError(f"Unknown activation function: {l}")
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Prevent overflow
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def categorical_cross_entropy(self, t, y, epsilon=1e-10):
        y = np.clip(y, epsilon, 1 - epsilon)
        return -np.sum(t * np.log(y))

    def relu(self,x):
        return np.maximum(0, x)
    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)
    def sigmoid(self,x):
        x = np.clip(x, -500, 500)  
        return 1 / (1 + np.exp(-x))
        