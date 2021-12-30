# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 13:57:37 2021

@author: jjerry yin
"""


import math
import matplotlib.pyplot as plt
#%% Problem 1
print('Problem 1:')
def combination(n, r):
    x = math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
    return x
    
pH0 = 0.5  # My hypothesis is that the coin is fair 
a_lis = [5, 10, 100, 100] 
b_lis = [3, 6, 60, 40]
p_value = 0
for j in range(len(a_lis)):   
    for i in range(b_lis[j],a_lis[j] + 1):
        p_value = p_value + (combination(a_lis[j],i) * (pH0 ** a_lis[j]))
    if p_value < 0.05:
        print('The p-value is %f < 0.05.\nThe evidence that out of %d flips %d were heads is\
              \nstrong enough to reject the hypothesis that the coin is fair.\n' % (p_value, a_lis[j], b_lis[j]))
    else:
        print('The p-value is %f >0.05.\nThe evidence that out of %d flips %d were heads is weak,\
              \nso we accept the hypothesis that the coin is fair.\n' % (p_value, a_lis[j], b_lis[j]))    
    p_value = 0

#%% Problem 2
print('Problem 2:')
x_lis = [0, 1, 2, 3, 4, 5, 6]
y_lis = [1, 2, 3, 4, 5, 6]
for y in y_lis:
    for x in x_lis:
        if x <= y:
            p = combination(y, x) * (0.5 ** x) * ((0.5) ** (y - x)) * (1 / 6)
            print('P[X=%d,Y=%d] = %f' % (x, y, p))
        else:
            break

#%% Problem 4
print('Problem 4:')
# calculate the conditional probability for manually making lists
x_lis = [0, 1, 2, 3, 4, 5, 6]
y_lis = [1, 2, 3, 4, 5, 6]
for y in y_lis:
    for x in x_lis:
        if x <= y:
            p = combination(y, x) * (0.5 ** x) * ((0.5) ** (y - x))
            print('P[X=%d|Y=%d] = %f' % (x, y, p))
        else:
            break
#%%
# I need to manually copy the results into different lists
Px0_lis=[0.5,0.25,0.125,0.0625,0.03125,0.015625]
Px1_lis=[0.5,0.5,0.375,0.25,0.15625,0.09375]
Px2_lis=[0.25,0.375,0.375,0.3125,0.234375]
Px3_lis=[0.125,0.25,0.3125,0.3125]
Px4_lis=[0.0625,0.15625,0.234375]
Px5_lis=[0.03125,0.09375]
Px6_lis=[0.015625]

lis = [Px0_lis,Px1_lis,Px2_lis,Px3_lis,Px4_lis,Px5_lis,Px6_lis]
P_x = 0
Px_lis = []
for li in lis:
    for i in range(len(li)):
        P_x = P_x + li[i] * (1/6)
    Px_lis.append(P_x)
    P_x = 0

# calculate mean and variance of X
EX = 0
EX2 = 0
for i in range(len(Px_lis)):
    EX = EX + Px_lis[i] * i
    EX2 = EX2 + Px_lis[i] * (i ** 2)
VarX = EX2 - (EX ** 2)
print('E[X] = %f, Var[X] = %f' % (EX,VarX))

# calculate mean and variance of Y
EY = 0
EY2 = 0
for j in range(1,7):
    EY = EY + j * (1/6)
    EY2 = EY2 + (j ** 2) * (1/6)
VarY = EY2 - (EY ** 2)
print('E[Y] = %f, Var[Y] = %f' % (EY,VarY))

#calculate rho
x_lis = [0, 1, 2, 3, 4, 5, 6]
y_lis = [1, 2, 3, 4, 5, 6]
EXY = 0
for y in y_lis:
    for x in x_lis:
        if x <= y:
            p = combination(y, x) * (0.5 ** x) * ((0.5) ** (y - x)) * (1 / 6)
            EXY = EXY + p * x * y
        else:
            break
rho = (EXY - EX * EY) / (math.sqrt(VarX * VarY))
print('E[XY] = %f, rho = %f' % (EXY, rho))

# calculate a* and b* for the liner estimator
a = rho * (math.sqrt(VarY / VarX))
b = EY - rho * (math.sqrt(VarY / VarX)) * EX
print('a*=%f, b*=%f' % (a,b))
print('The linear predictor is: f(X)=%fX+%f '% (a,b))

# prediction of Y give X=0,1,...,6
Y_liner_list = []
for xx in range(0,7):
    Y = a * xx + b
    Y_liner_list.append(Y)
    print('When X=%d, Y=%f' % (xx, Y))


#%% plot for comparison
Y_01loss = [1,2,3,5,6,6,6]
plt.figure()
plt.scatter(x_lis,Y_01loss) # 0-1 predictor in Problem 2
plt.scatter(x_lis,Y_liner_list) # liner predictor in Problem 4
plt.legend(['0-1 predictor in Problem 2', 'liner predictor in Problem 4'], loc=2)
plt.title('Comparison')
plt.xlabel('X value'),plt.ylabel('predicted value of Y')





