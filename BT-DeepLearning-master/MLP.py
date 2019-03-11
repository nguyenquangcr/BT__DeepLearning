#-------------------------------------------------------------------------------
#STEP 01: TẠO DỮ LIỆU GIẢ
import math
import numpy as np
import matplotlib.pyplot as plt

N = 200 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d0, N*C)) # data matrix (each row = single example)
y = np.zeros(N*C, dtype='uint8') # class labels

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.4 # theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j

# lets visualize the data:
plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7);

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.show()

#-------------------------------------------------------------------------------
#STEP 02: ĐỊNH NGHĨA CÁC HÀM BỔ TRỢ
## Hàm nhập vào số layer ẩn và số unit mỗi layer ẩn đó
def enter_variable():
    a = input("Nhap so hidden layer (1 or 2 or 3): ")
    hl = int(a)
    d = [0,0,0]
    for i in range(hl):
        d[i] = input("Nhap so unit cua hidden laye thu {0}: ".format(i))
    n = int(input("Nhap so vong lap training MLP: "))
    return hl,d,n

## Softmax function
def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-hot coding
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# cost or loss function use L2 Regularization (for 1/2/3 hidden layer)
l = 0.01
def cost1(Y, Yhat, W1, W2):
    w1 = np.square(W1).sum()
    w2 = np.square(W2).sum()
    R = w1 + w2
    return (-np.sum(Y*np.log(Yhat))/Y.shape[1])+(l/(2*Y.shape[1]))*R
def cost2(Y, Yhat, W1, W2, W3):
    w1 = np.square(W1).sum()
    w2 = np.square(W2).sum()
    w3 = np.square(W3).sum()
    R = w1 + w2 + w3
    return (-np.sum(Y*np.log(Yhat))/Y.shape[1])+(l/(2*Y.shape[1]))*R
def cost3(Y, Yhat, W1, W2, W3, W4):
    w1 = np.square(W1).sum()
    w2 = np.square(W2).sum()
    w3 = np.square(W3).sum()
    w4 = np.square(W4).sum()
    R = w1 + w2 + w3 + w4
    return (-np.sum(Y*np.log(Yhat))/Y.shape[1])+(l/(2*Y.shape[1]))*R

#-------------------------------------------------------------------------------
#STEP 03: HUẤN LUYỆN MẠNG VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC TRÊN TẬP TRAINING
#region HUẤN LUYỆN MẠNG
hl, d , n = enter_variable()
print("So luong hidden layer la : {0}".format(hl))
for i in range(hl):
    print("So unit cua hidden laye thu {0}: {1}".format(i,d[i]))
print("So vong lap training MLP la : {0}".format(n))
#các biến chung
d0 = 2           # size of input layer
d1 = int(d[0])   # size of hidden layer 1
d2 = int(d[1])   # size of hidden layer 2
d3 = int(d[2])   # size of hidden layer 3
d_output = C = 3    #size of hidden layer
Y = convert_labels(y, C) #one hot coding for labels
N = X.shape[1]
eta = 1 # learning rate

# Huấn luyện và đánh giá độ chính xác MLP 1/2/3 hidden layer
def TrainMLP_1hiddenlayer():
    d2 = d_output
    # initialize parameters randomly
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
    # training
    for i in range(n):
        ## Feedforward
        Z1 = np.dot(W1.T, X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2.T, A1) + b2
        Yhat = softmax(Z2)

        # print loss after each 1000 iterations
        if i %1000 == 0:
            # compute the loss: average cross-entropy loss
            loss = cost1(Y, Yhat, W1,W2)
            print("iter %d, loss: %f" %(i, loss))

        # backpropagation
        E2 = (Yhat - Y )/N
        dW2 = np.dot(A1, E2.T)
        db2 = np.sum(E2, axis = 1, keepdims = True)
        E1 = np.dot(W2, E2)
        E1[Z1 <= 0] = 0 # gradient of ReLU
        dW1 = np.dot(X, E1.T)
        db1 = np.sum(E1, axis = 1, keepdims = True)

        # Gradient Descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
    #Đánh giá độ chính xác
    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    predicted_class = np.argmax(Z2, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.8f %%' % acc)
    return 0
def TrainMLP_2hiddenlayer():
    d3 = d_output
    # initialize parameters randomly
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
    W3 = 0.01*np.random.randn(d2, d3)
    b3 = np.zeros((d3, 1))
    #Training
    for i in range(n):
        ## Feedforward
        Z1 = np.dot(W1.T, X) + b1
        A1 = np.maximum(Z1, 0) #ReLU
        Z2 = np.dot(W2.T, A1) + b2
        A2 = np.maximum(Z2, 0) #ReLU
        Z3 = np.dot(W3.T, A2) + b3

        Yhat = softmax(Z3) #Softmax for output

        # print loss after each 1000 iterations
        if i %1000 == 0:
            # compute the loss: average cross-entropy loss
            loss = cost2(Y, Yhat, W1, W2, W3)
            print("iter %d, loss: %f" %(i, loss))

        # backpropagation
    
        E3 = (Yhat - Y )/N
        dW3 = np.dot(A2, E3.T)
        db3 = np.sum(E3, axis = 1, keepdims = True)
        E2 = np.dot(W3, E3)
        E2[Z2 <= 0] = 0 # gradient of ReLU
        dW2 = np.dot(A1, E2.T)
        db2 = np.sum(E2, axis = 1, keepdims = True)
        E1 = np.dot(W2, E2)
        E1[Z1 <= 0] = 0 # gradient of ReLU
        dW1 = np.dot(X, E1.T)
        db1 = np.sum(E1, axis = 1, keepdims = True)

        # Gradient Descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
        W3 += -eta*dW3
        b3 += -eta*db3
    #Đánh giá độ chính xác
    Z1 = np.dot(W1.T, X) + b1 
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.maximum(Z2, 0)
    Z3 = np.dot(W3.T, A2) + b3

    predicted_class = np.argmax(Z3, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.8f %%' % acc)
    return 0
def TrainMLP_3hiddenlayer():
    d4 = d_output
    # initialize parameters randomly
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
    W3 = 0.01*np.random.randn(d2, d3)
    b3 = np.zeros((d3, 1))
    W4 = 0.01*np.random.randn(d3, d4)
    b4 = np.zeros((d4, 1))
    #Training
    for i in range(n):
        ## Feedforward
        Z1 = np.dot(W1.T, X) + b1
        A1 = np.maximum(Z1, 0) #ReLU
        Z2 = np.dot(W2.T, A1) + b2
        A2 = np.maximum(Z2, 0) #ReLU
        Z3 = np.dot(W3.T, A2) + b3
        A3 = np.maximum(Z3, 0)
        Z4 = np.dot(W4.T, A3) + b4
        Yhat = softmax(Z4) #Softmax for output

        # print loss after each 1000 iterations
        if i %1000 == 0:
            # compute the loss: average cross-entropy loss
            loss = cost3(Y, Yhat, W1, W2, W3, W4)
            print("iter %d, loss: %f" %(i, loss))

        # backpropagation
        E4 = (Yhat - Y )/N
        dW4 = np.dot(A3, E4.T)
        db4 = np.sum(E4, axis = 1, keepdims = True)
        E3 = np.dot(W4, E4)
        E3[Z3 <= 0] = 0 # gradient of ReLU
        dW3 = np.dot(A2, E3.T)
        db3 = np.sum(E3, axis = 1, keepdims = True)
        E2 = np.dot(W3, E3)
        E2[Z2 <= 0] = 0 # gradient of ReLU
        dW2 = np.dot(A1, E2.T)
        db2 = np.sum(E2, axis = 1, keepdims = True)
        E1 = np.dot(W2, E2)
        E1[Z1 <= 0] = 0 # gradient of ReLU
        dW1 = np.dot(X, E1.T)
        db1 = np.sum(E1, axis = 1, keepdims = True)

        # Gradient Descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
        W3 += -eta*dW3
        b3 += -eta*db3
        W4 += -eta*dW4
        b4 += -eta*db4

    #Đánh giá độ chính xác
    Z1 = np.dot(W1.T, X) + b1 
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.maximum(Z2, 0)
    Z3 = np.dot(W3.T, A2) + b3
    A3 = np.maximum(Z3, 0)
    Z4 = np.dot(W4.T, A3) + b4

    predicted_class = np.argmax(Z4, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.8f %%' % acc)
    return 0

# Xét từng trường hợp 1,2,3 hidden layer 
if(hl==1):          #1 hidden layer
    TrainMLP_1hiddenlayer()
elif (hl==2):       #2 hidden layer
    TrainMLP_2hiddenlayer()
else:               #3 hidden layer
    TrainMLP_3hiddenlayer()
#endregion
#-------------------------------------------------------------------------------
