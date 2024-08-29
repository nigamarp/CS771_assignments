import numpy as np
#Loading the given Dataset
X_seen=np.load('Dataset\X_seen.npy',allow_pickle=True,encoding='latin1')
Xtest=np.load('Dataset/Xtest.npy',allow_pickle=True,encoding='latin1')
Ytest=np.load('Dataset/Ytest.npy',allow_pickle=True,encoding='latin1')
class_attributes_seen=np.load('Dataset/class_attributes_seen.npy',allow_pickle=True,encoding='latin1')
class_attributes_unseen=np.load('Dataset/class_attributes_unseen.npy',allow_pickle=True,encoding='latin1')
X_Mean=[np.mean(X_seen[i], axis=0) for i in range(40)] #Calculation the mean of the seen classes using the training data given
As=class_attributes_seen #attribute matrix of seen classes(1-40)
AsTAs= np.dot(As.T, As) #To calculate W
AsTMs=np.dot(As.T,X_Mean)#To calculate W
Lambda=[0.01, 0.1, 1, 10, 20, 50, 100]
row,column=AsTAs.shape
I= np.eye(row)  #Identity Matrix
MaxAccuracy=0.0
Lambda_MaxAccuracy=0.0
for i in Lambda:
    A=np.linalg.inv((AsTAs+i*I)) #Here (A=AsTAs+Lambda*I)^-1
    W=np.dot(A,AsTMs)
    # print(W)
    Mean_unseen=[]
    for j in range(10):
        temp=np.dot(W.T,class_attributes_unseen[j]) #Calculating the mean of unseen classes
        Mean_unseen.append(temp)
    Predictions=[]
    for X_n in Xtest:
        d={}
        for k in range(10):   #Finding out Euclidean Distances from each of 10 unseen classes
            d[k]=np.linalg.norm(Mean_unseen[k] - X_n)   
        Predictions.append(min(d, key=d.get)+1)  #Prediction is the closest class
    Correct=0
    Y_n=0
    for j in Ytest:
        if Predictions[Y_n]==j:Correct+=1
        Y_n+=1
    accuracy=Correct/len(Predictions)
    print(i,"-",accuracy)
    print()
    if accuracy>MaxAccuracy:
        MaxAccuracy=accuracy
        Lambda_MaxAccuracy=i
    # A[]=accuracy
# print(A)
# print(max(A,key=A.get))
print(Lambda_MaxAccuracy,"  -  ",MaxAccuracy)

