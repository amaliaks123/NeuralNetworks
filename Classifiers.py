from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar100
import numpy as np

(xtrain, ytrain), (xtest, ytest) = cifar100.load_data()
xtrain = xtrain.reshape(xtrain.shape[0], -1).astype('float32')
xtest = xtest.reshape(xtest.shape[0], -1).astype('float32')

#regularization of data
xtrain = xtrain / 255
xtest = xtest / 255

pca = PCA(n_components=70)
xtrain_pca = pca.fit_transform(xtrain)
xtest_pca = pca.transform(xtest)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(xtrain_pca, ytrain.ravel())

ypred = knn3.predict(xtest_pca)
accuracy = accuracy_score(ytest, ypred)
print("KNN accuracy for k=3:", accuracy)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar100
import numpy as np

(xtrain, ytrain), (xtest, ytest) = cifar100.load_data()
xtrain = xtrain.reshape(xtrain.shape[0], -1).astype('float32')
xtest = xtest.reshape(xtest.shape[0], -1).astype('float32')

#regularization of data
xtrain = xtrain / 255
xtest = xtest / 255

pca = PCA(n_components=70)
xtrain_pca = pca.fit_transform(xtrain)
xtest_pca = pca.transform(xtest)

#k-NN Classifier for k=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(xtrain_pca, ytrain.ravel())

ypred = knn1.predict(xtest_pca)
accuracy = accuracy_score(ytest, ypred)
print("KNN accuracy for k=1:", accuracy)

from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar100
import numpy as np

(xtrain, ytrain), (xtest, ytest) = cifar100.load_data()
xtrain = xtrain.reshape(xtrain.shape[0], -1).astype('float32')
xtest = xtest.reshape(xtest.shape[0], -1).astype('float32')

#regularization of data
xtrain = xtrain / 255
xtest = xtest / 255

pca = PCA(n_components=70)
xtrain_pca = pca.fit_transform(xtrain)
xtest_pca = pca.transform(xtest)

#Nearest Centroid Classifier
nc = NearestCentroid()
nc.fit(xtrain_pca, ytrain.ravel())

ypred = nc.predict(xtest_pca)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy for nearest centroid:", accuracy)
