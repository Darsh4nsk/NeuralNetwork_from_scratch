import os 
import numpy as np 
import nn
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist # type: ignore

def onehot(y):
    a = []
    for i in y:
        y = np.zeros((10,1))
        y[i] = [1]
        a.append(y)
    return np.array(a)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
temp_xtest = x_test
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1] * x_train.shape[2],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1] * x_test.shape[2],1))

x_train = x_train/np.max(x_train)
x_test = x_test/np.max(x_train)

y_train = onehot(y_train)
y_test = onehot(y_test)


np.set_printoptions(precision=4,suppress=True)
network = nn.Network([784,20,10,10],['r','r','sof'])
his = network.training(x_train,y_train,0.01,200,10000) #inputs: training x, y, learning_rate,epoch, total_training_num
plt.plot(his)
plt.show()
n = int(input("enter"))
predicted_label = np.argmax(network.forward_feed(x_test[n]))
actual_label = np.argmax(y_test[n])
print(f"Predicted: {predicted_label}, Actual: {actual_label}")
plt.imshow(temp_xtest[n])
plt.show()
print(network.testing(x_test,y_test)) #inputs: testing data x, y 