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

def kernel(x_n,x_m):
    dist=sum((x - y) ** 2 for x, y in zip(x_n,x_m))
    # dist = (np.linalg.norm(x_n - x_m))**2
    dist = dist*(-0.1)
    # print(math.exp(dist))
    return math.exp(dist)
n_o_ls=[1]
for j in range(50):
    for n_o_l in n_o_ls:
        landmarks=[]
        for i in range(n_o_l):
            index=random.randint(0,len(Train)-1)
            landmarks.append(Train[index])
            x=Train[index][0]
            y=Train[index][1]
        New_Training_set=[[0,0] for i in range(len(Train))]
        for i in range(len(Train)):
            for j in range(n_o_l):
                New_Training_set[i][0]=(kernel(Train[i],landmarks[j]))
        u=[]
        u.append(New_Training_set[0][0])
        u.append(New_Training_set[1][0])
        for j in range(20):
            U1=0
            count1=0
            U2=0
            count2=0
            for i in range(len(Train)):
                if np.linalg.norm(u[0]-New_Training_set[i][0])<np.linalg.norm(u[1]-New_Training_set[i][0]):
                    U1+=New_Training_set[i][0]
                    New_Training_set[i][1]=0
                    count1+=1
                    # print(i,"#")
                else:
                    U2+=New_Training_set[i][0]
                    New_Training_set[i][1]=1
                    count2+=1
                    # print(i,"$")
            u[0]=U1/count1
            u[1]=U2/count2
        # X=[]
        # Y=[]
        # for i in range(len(Train)):
        #     X.append(New_Training_set[i][0])
        #     Y.append(Train[i][1])
        # plt.scatter(X,Y,c="blue")
        Y1=[]
        X1=[]
        X2=[]
        Y2=[]
        for i in range(2,len(Train)):
            if New_Training_set[i][1]==0 and Train[i][0]!=x and Train[i][1]!=y:
                Y1.append(Train[i][1])
                X1.append(Train[i][0])
            elif Train[i][0]!=x and Train[i][1]!=y:
                Y2.append(Train[i][1])
                X2.append(Train[i][0])
        plt.scatter(X1, Y1, c ="red")
        plt.scatter(X2, Y2, c ="green")
        plt.scatter(x,y,c="blue")
        # plt.scatter(Train[0][0],Train[0][1],c="black")
        # plt.scatter(Train[1][0],Train[1][1],c="brown")
        plt.show()    