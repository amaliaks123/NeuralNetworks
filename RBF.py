# PCA, SMOTE, sigma tuning and weight regularization

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.datasets import cifar10
import numpy as np
import time

class RBF:
    def __init__(self, num_centers):
        self.num_centers = num_centers
        self.centers = None
        self.weights = None
        self.sigma = None

    def _rbf(self, x, center, sigma):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def _calculate_activations(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = self._rbf(x, center, self.sigma)
        return G

    def fit(self, X, y, reg_param=0.01):
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Dynamically calculate sigma 
        self.sigma = np.std([np.linalg.norm(center - other_center)
                             for center in self.centers for other_center in self.centers])
        G = self._calculate_activations(X)

        # Regularized weights
        self.weights = np.linalg.pinv(G.T @ G + reg_param * np.eye(G.shape[1])) @ G.T @ y

    def predict(self, X):
        G = self._calculate_activations(X)
        predictions = G.dot(self.weights)
        return np.argmax(predictions, axis=1)

# Data preparation and PCA
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_0, class_1 = 0, 1
train_mask = (y_train.flatten() == class_0) | (y_train.flatten() == class_1)
test_mask = (y_test.flatten() == class_0) | (y_test.flatten() == class_1)

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

y_train = (y_train.flatten() == class_1).astype(int)
y_test = (y_test.flatten() == class_1).astype(int)

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Cross-validation
hidden_neurons_list = [50, 70, 100, 150, 200]
k_folds = 5
results = []
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for num_centers in hidden_neurons_list:
    print(f"Evaluating RBF Network with {num_centers} neurons...")
    fold_results = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        y_train_oh = np.eye(2)[y_train_fold]
        y_val_oh = np.eye(2)[y_val_fold]

        rbf_net = RBF(num_centers=num_centers)
        rbf_net.fit(X_train_fold, y_train_oh)
        y_val_pred = rbf_net.predict(X_val_fold)
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        fold_results.append(val_acc)

    mean_val_acc = np.mean(fold_results)
    print(f"Neurons: {num_centers}, Mean Validation Accuracy: {mean_val_acc:.2f}")
    results.append((num_centers, mean_val_acc))

# Evaluate on test set
best_num_centers = max(results, key=lambda x: x[1])[0]
rbf_net = RBF(num_centers=best_num_centers)
rbf_net.fit(X_train, np.eye(2)[y_train])
y_test_pred = rbf_net.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Best Neurons: {best_num_centers}, Test Accuracy: {test_acc:.2f}")







# Modified RBF class with sigma tuning and weight regularization

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from tensorflow.keras.datasets import cifar10
import numpy as np
import time


class RBF:
    def __init__(self, num_centers):
        self.num_centers = num_centers
        self.centers = None
        self.weights = None
        self.sigma = None

    def _rbf(self, x, center, sigma):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def _calculate_activations(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = self._rbf(x, center, self.sigma)
        return G

    def fit(self, X, y, reg_param=0.01):
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Dynamically calculate sigma based on cluster variance
        self.sigma = np.std([np.linalg.norm(center - other_center)
                             for center in self.centers for other_center in self.centers])
        G = self._calculate_activations(X)

        # Regularized weights
        self.weights = np.linalg.pinv(G.T @ G + reg_param * np.eye(G.shape[1])) @ G.T @ y

    def predict(self, X):
        G = self._calculate_activations(X)
        predictions = G.dot(self.weights)
        return np.argmax(predictions, axis=1)

# Data preparation and PCA
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_0, class_1 = 0, 1
train_mask = (y_train.flatten() == class_0) | (y_train.flatten() == class_1)
test_mask = (y_test.flatten() == class_0) | (y_test.flatten() == class_1)

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

y_train = (y_train.flatten() == class_1).astype(int)
y_test = (y_test.flatten() == class_1).astype(int)

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Cross-validation
hidden_neurons_list = [50, 70, 100, 150, 200]
k_folds = 5
results = []
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for num_centers in hidden_neurons_list:
    print(f"Evaluating RBF Network with {num_centers} neurons...")
    fold_results = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        y_train_oh = np.eye(2)[y_train_fold]
        y_val_oh = np.eye(2)[y_val_fold]

        rbf_net = RBF(num_centers=num_centers)
        rbf_net.fit(X_train_fold, y_train_oh)
        y_val_pred = rbf_net.predict(X_val_fold)
        val_acc = accuracy_score(y_val_fold, y_val_pred)
        fold_results.append(val_acc)

    mean_val_acc = np.mean(fold_results)
    print(f"Neurons: {num_centers}, Mean Validation Accuracy: {mean_val_acc:.2f}")
    results.append((num_centers, mean_val_acc))

# Evaluate on test set
best_num_centers = max(results, key=lambda x: x[1])[0]
rbf_net = RBF(num_centers=best_num_centers)
rbf_net.fit(X_train, np.eye(2)[y_train])
y_test_pred = rbf_net.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Best Neurons: {best_num_centers}, Test Accuracy: {test_acc:.2f}")



