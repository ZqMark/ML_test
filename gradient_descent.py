from numpy import *

# 数据集大小
m = 20
#
X0 = ones((m,1)) # 生成一个m行一列的向量,全是１
print('x0:',X0)
X1 = arange(1,m+1).reshape(m,1)
print('x1:',X1)
X = hstack((X0,X1)) # 按照列堆叠形成数组,样本数据
print('X:',X)

#
Y = array([
    3,4,5,5,2,5,7,8,11,8,12,
    11,13,13,16,17,18,17,19,21

]).reshape(m,1)
print('Y:',Y)
# 学习率
alpha = 0.01

# 定义代价函数
def cost_function(theta,X,Y):
    diff = dot(X,theta) - Y  # dot() 数组需要像举证那样相乘,就需要用到dot()
    return (1/(2*m))*dot(diff.transpose(),diff)

# 定义代价函数对应的梯度函数
def gradient_function(theta,X,Y):
    diff = dot(X,theta) - Y  # dot:矩阵相乘
    return (1/m)*dot(X.transpose(),diff)

#梯度下降迭代
def gradient_descent(X,Y,alpha):
    theta = array([1,1]).reshape(2,1)
    print('theta:',theta)
    gradient = gradient_function(theta,X,Y)
    while not all(abs(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta,X,Y)
    return theta

optimal = gradient_descent(X,Y,alpha)
print('optimal:',optimal)
print('cost function:',cost_function(optimal,X,Y)[0][0])

# 根据数据画出对应的图像
def plot(X,Y,theta):
    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    ax.scatter(X,Y,s= 30, c = "red",marker="s")
    plt.xlabel("X")
    plt.ylabel("Y")
    x = arange(0,21,0.2) # x的范围
    y = theta[0] + theta[1]*x
    ax.plot(x,y)
    plt.show()

plot(X1,Y,optimal)

