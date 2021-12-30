# -*- coding: utf-8 -*-
"""
Created on Fri Nov 5 6:14:22 2021

@author: Jerry Yin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def pick_N_img_subset(X,N):
    Y_label = X[0]
    np.random.seed(2)
    j = np.random.randint(0,len(X)-N)
    X_train_s = X.iloc[j:j+N,1:]/255
    Y_label_s = Y_label.iloc[j:j+N]    
    return X_train_s, Y_label_s.astype('float64')



def target_cloth(Y,fashion_num):
    Y = np.array(Y)
    Y_modified = Y
    for i in range(len(Y)):
        if Y_modified[i] == fashion_num:
            Y_modified[i] = 1
        else:
            Y_modified[i] = 0
    return Y_modified



def sigmoid(z):
    return np.exp(-z) / (1.0 + np.exp(-z))



def SGD(X,Y,itera,step):
    theta = np.zeros((X.shape[1]))
    t_lis = []
    np.random.seed(2)
    start = time.time()
    for t in range(1, itera+1):
        t_lis.append(t)
        i = np.random.randint(0,len(X))
        G = (Y[i] + sigmoid(np.dot(X[i],theta))) * X[i]
        theta_new = theta - G/(t ** (step))
    Time = time.time() - start
    theta_final = theta_new
    return theta_final, Time



def classifier(X,Y,b):
    mul = -np.dot(X,b)
    exp = np.exp(mul)
    P_y = 1/(1 + exp)
    Y_hat_lis = []
    Y_hat = 0
    for p in P_y:
        if p >= 0.5:
            Y_hat = 1
            Y_hat_lis.append(Y_hat)
        if 0 <= p < 0.5:
            Y_hat = 0
            Y_hat_lis.append(Y_hat)
    return Y_hat_lis



def Accuracy(Y_hat,Y):
    count = 0
    Y = np.array(Y)
    E_lis = []
    for i in range(len(Y_hat)):
        E_lis.append(int(Y[i] - Y_hat[i]))
        if Y_hat[i] == Y[i]:
            count += 1
        else:
            continue
    accur = count / len(Y_hat)
    return accur,E_lis



# whole data set (60000,785) including label col
train = pd.read_csv('fashion-mnist_train.csv', header=None, skiprows=1)
test = pd.read_csv('fashion-mnist_test.csv', header=None, skiprows=1)

fashion_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal'
                 ,'Shirt','Sneaker','Bag','Ankle boot']

fashion_num_lis = [0,1,2,3,4,5,6,7,8,9]

num_training = 1000 # randomly pick 1000 training data
num_testing = 500 # randomly pick 500 testing data
epoches = 300 # 300 epoches for training

#%%
# Display first 30 data in Fashion MINIST data set
refined_train = train.T[1:785]
labels = train[0]
plt.figure(1)
rows = 5
cols = 6
for i in range(rows * cols):
    plt.subplot(rows,cols,i+1)
    plt.title('%s, label=%d' % (fashion_names[labels[i]],labels[i]))
    src = np.reshape(np.array(refined_train[i]),(28,28))
    plt.imshow(src,cmap='gray')

#%%
print('--------------Running SGD--------------')
a_SGD = []
time_SGD = []


for fashion_num in fashion_num_lis:
    
    #Training
    X_small_train,Y_small_train = pick_N_img_subset(train,num_training) 
    Y_small_train_0_1s = target_cloth(Y_small_train,fashion_num)
    insert_1s_train = np.ones((X_small_train.shape[0], 1))
    X_small_train_0_1s = np.hstack((insert_1s_train, X_small_train))
    Theta, tt= SGD(X_small_train_0_1s,Y_small_train_0_1s,itera=epoches,step=3/4)
    time_SGD.append(tt)
    
    #Testing
    X_small_test,Y_small_test = pick_N_img_subset(test,num_testing)
    Y_small_test_0_1s = target_cloth(Y_small_test,fashion_num)
    insert_1s_test = np.ones((X_small_test.shape[0], 1))
    X_small_test_0_1s = np.hstack((insert_1s_test, X_small_test))   
    Y_hats = classifier(X_small_test_0_1s, Y_small_test_0_1s,Theta)
    a, Expectations = Accuracy(Y_hats,Y_small_test_0_1s)
    wrong = len(Expectations) -  Expectations.count(0)
    a_SGD.append(a)
    
    print('Accuracy of classification of %s is %.6f' % (fashion_names[fashion_num],a))
    print('Wrongly classified amount: %d/%d' % (wrong,num_testing))
    print('Time consumed: %.6f s\n' % tt)


#%%
print('--------------Running GD--------------')
# Traditional GD for comparision
def Gradient_Descent(X,Y,sigma2,step,N):
    betas = np.zeros((X.shape[1]))
    summ = betas
    itera = 0
    itera_lis = []
    loss_lis = []
    m = X.shape[0]
    start = time.time()
    for itera in range(N):   
        itera_lis.append(itera)        
        for i in range(m):
            summ = summ + (Y[i] - 1 + sigmoid(np.dot(X[i],betas))) * X[i]
        G = summ - (1/sigma2) * betas    
        loss_lis.append(np.abs(np.mean(G)))   
        betas_new = betas + step * G   
        betas = betas_new
    Time = time.time() - start
    betas_final = betas
    return itera_lis, loss_lis, betas_final, Time

a_GD = []
time_GD = []
loss_lis_GD = []
iter_lis_GD = []
for fashion_num in fashion_num_lis:
    
    #Training
    X_small_train,Y_small_train = pick_N_img_subset(train,num_training) 
    Y_small_train_0_1s = target_cloth(Y_small_train,fashion_num)
    insert_1s_train = np.ones((X_small_train.shape[0], 1))
    X_small_train_0_1s = np.hstack((insert_1s_train, X_small_train))
    iters, loss, Theta_GD, tt2 = Gradient_Descent(X_small_train_0_1s,Y_small_train_0_1s,sigma2=0.01,step=0.001,N=epoches) 
    time_GD.append(tt2)
    loss_lis_GD.append(loss)
    iter_lis_GD.append(iters)
    #Testing
    X_small_test,Y_small_test = pick_N_img_subset(test,num_testing) 
    Y_small_test_0_1s = target_cloth(Y_small_test,fashion_num)
    insert_1s_test = np.ones((X_small_test.shape[0], 1))
    X_small_test_0_1s = np.hstack((insert_1s_test, X_small_test))   
    Y_hats = classifier(X_small_test_0_1s, Y_small_test_0_1s,Theta_GD)
    a2, Expectations2 = Accuracy(Y_hats,Y_small_test_0_1s)
    wrong2 = len(Expectations2) -  Expectations2.count(0)
    a_GD.append(a2)
    
    print('Accuracy of classification of %s is %.6f' % (fashion_names[fashion_num],a2))
    print('Wrongly classified amount: %d/%d' % (wrong2,num_testing))
    print('Time consumed: %.6f s\n' % tt2)


#%%
# Plot results of SGD
plt.figure(2)
plt.subplot(1,2,1)
plt.title('Accuracy with SGD')
plt.scatter(fashion_num_lis, a_SGD)
plt.xlabel('Fashion label #'),plt.ylabel('Accuracy')
plt.xticks(fashion_num_lis,fashion_names,rotation=45)
plt.yticks(np.arange(0.85, 1, 0.025))
plt.subplot(1,2,2)
plt.title('Time consuming with SGD')
plt.scatter(fashion_num_lis, time_SGD,c='r')
plt.xlabel('Fashion label #'),plt.ylabel('Consumed time (s)')
plt.xticks(fashion_num_lis,fashion_names,rotation=45)

# Plot results of GD
plt.figure(3)
plt.subplot(1,2,1)
plt.title('Accuracy with GD')
plt.scatter(fashion_num_lis, a_GD)
plt.xlabel('Fashion label #'),plt.ylabel('Accuracy')
plt.xticks(fashion_num_lis,fashion_names,rotation=45)
plt.yticks(np.arange(0.85, 1, 0.025))
plt.subplot(1,2,2)
plt.title('Time consuming with GD')
plt.scatter(fashion_num_lis, time_GD,c='r')
plt.xlabel('Fashion label #'),plt.ylabel('Consumed time (s)')
plt.xticks(fashion_num_lis,fashion_names,rotation=45)

#%%

plt.figure(4)
r, c = 2, 5
for i in range(r*c):
    plt.subplot(r,c, i+1)
    plt.title('Loss of classifying %s' % fashion_names[i])
    plt.plot(iter_lis_GD[i],loss_lis_GD[i])
    plt.xlabel('Epochs #'),plt.ylabel('Loss #')
    
    
    
    
    

