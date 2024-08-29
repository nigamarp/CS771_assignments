from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
# X=np.load('C:/Users/hp/Desktop/ML/mnist_small.pkl',allow_pickle=True,encoding='latin1')
with open('C:/Users/hp/Desktop/IIT Slides/Machine Learning/Assignment_2/mnist_small.pkl', 'rb') as file:
    # Load the object from the file
    X = pickle.load(file)
Y=X['Y']
X=X['X']
# print(Y)
# print(Y)
# print(len(X))
pca = PCA(n_components=2)
tsne = TSNE(n_components=2) 
X_pca = pca.fit_transform(X)
X_embedded = tsne.fit_transform(X)
color=['blue','green','red','black','yellow','purple','orange','pink','brown','gray']
X1=[[] for i in range(10)]
Y1=[[] for i in range(10)]
for i in range(len(X_pca)):
    X1[Y[i][0]].append(X_pca[i][0])
    Y1[Y[i][0]].append(X_pca[i][1])
for i in range(10):
    plt.scatter(X1[i],Y1[i], c=color[i])
plt.title("P.C.A")
# print(Y[10])
plt.show()
X2=[[] for i in range(10)]
Y2=[[] for i in range(10)]
for i in range(len(X_embedded)):
    X2[Y[i][0]].append(X_embedded[i][0])
    Y2[Y[i][0]].append(X_embedded[i][1])
for i in range(10):
    plt.scatter(X2[i],Y2[i], c=color[i])
plt.title("T.S.N.E")
plt.show()