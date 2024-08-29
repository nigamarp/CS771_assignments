import numpy as np
import math
import matplotlib.pyplot as plt
import random

f = open("C:/Users/hp/Desktop/IIT Slides/Machine Learning/Assignment_2/kmeans_data.txt", "r")
data=f.read()
Data_set=data.split()
Train=[]
for i in range(0, len(Data_set), 2):
    temp=[]
    temp.append(float(Data_set[i]))
    temp.append(float(Data_set[i+1]))
    Train.append(temp)
def phi(x_n,y_n):
    return math.sqrt(x_n**2+y_n**2)
data_points=[[0,0] for i in range(len(Train))]
for i in range(len(Train)):
    data_points[i][0]=(phi(Train[i][0],Train[i][1]))
u=[]
u.append(data_points[0][0])
u.append(data_points[1][0])
# print(u)
for j in range(20):
    U1=0
    count1=0
    U2=0
    count2=0
    for i in range(len(Train)):
        if np.linalg.norm(u[0]-data_points[i][0])<np.linalg.norm(u[1]-data_points[i][0]):
            U1+=data_points[i][0]
            data_points[i][1]=0
            count1+=1
            # print(i,"#")
        else:
            U2+=data_points[i][0]
            data_points[i][1]=1
            count2+=1
            # print(i,"$")
    # print(u)
    u[0]=U1/count1
    u[1]=U2/count2
    # print(u)
# Scatter plot in New Mapping
Y1=[]
X1=[]
X2=[]
Y2=[]
for i in range(len(Train)):
    if data_points[i][1]==0:
        Y1.append(Train[i][1])
        X1.append(Train[i][0])
    else:
        Y2.append(Train[i][1])
        X2.append(Train[i][0])
plt.scatter(X1, Y1, c ="red")
plt.scatter(X2, Y2, c ="green")
plt.show()