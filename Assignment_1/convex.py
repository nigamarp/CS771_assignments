import numpy as np
X_seen=np.load('Dataset/X_seen.npy',allow_pickle=True,encoding='latin1')
Xtest=np.load('Dataset/Xtest.npy',allow_pickle=True,encoding='latin1')
Ytest=np.load('Dataset/Ytest.npy',allow_pickle=True,encoding='latin1')
class_attributes_seen=np.load('Dataset/class_attributes_seen.npy',allow_pickle=True,encoding='latin1')
class_attributes_unseen=np.load('Dataset/class_attributes_unseen.npy',allow_pickle=True,encoding='latin1')
X_Mean=[np.mean(X_seen[i], axis=0) for i in range(40)]   #Mean of all the 40 seen class's test data
# print(X_Mean)
S=[]
for j in range(10):
    temp=[]
    for i in range(40):
        temp.append(np.dot(class_attributes_unseen[j],class_attributes_seen[i]))
    S.append(temp)
# S=[([np.dot(class_attributes_unseen[j],class_attributes_seen[i])]for i in range(40) )for j in range(10)]
# print(S)
for i in range(10):  #Normalization
    sum=0
    for j in range(40):
        sum+=S[i][j]
    for j in range(40):
        S[i][j]=S[i][j]/sum

# print(S)
#****************************************************************************************
Mean_unseen=[]
for j in range(10):
    t=[]
    for i in range(40):
        t.append(S[j][i]*X_Mean[i])
        # print(X_Mean[i],end="$")
        # print(t,end="$")
    # print(t)
    for k in range(1,40):
        for i in range(len(t[0])):
            t[0][i]+=t[k][i]
    Mean_unseen.append(t[0])
#print(np.shape(Mean_unseen))
#print(Mean_unseen)
#*****************************************************************************************
# Mean_unseen=[((S[j][i]*X_Mean[i]) for i in range(40)) for j in range(10)]
#print(Mean_unseen)
Predictions=[]
for X_n in Xtest:
    d={}
    for i in range(10):
        d[i]=np.linalg.norm(Mean_unseen[i] - X_n)
    Predictions.append(min(d, key=d.get)+1)
Correct=0
Y_n=0
# print(Predictions)
for j in Ytest:
    if Predictions[Y_n]==j:Correct+=1
    Y_n+=1
accuracy=Correct/len(Predictions)
print(accuracy)
# print(Correct)
