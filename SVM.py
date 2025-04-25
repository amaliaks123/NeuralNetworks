# RBF -- PCA -- HYPERPARAMETER SEARCH -- EXAMPLES OF CORRECT/WRONG CLASSIFICATION 

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import time

# Start measuring execution time
total_start_time = time.time()

# Step 1 : Load data
start_time = time.time()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Select two classes (e.g., 3: cat, 5: dog)
classes = [3, 5]
train_mask = np.isin(y_train, classes).flatten()
test_mask = np.isin(y_test, classes).flatten()

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# Convert labels to -1 and +1
y_train = np.where(y_train == classes[0], -1, 1).flatten()
y_test = np.where(y_test == classes[0], -1, 1).flatten()

# Flatten images and normalize
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

# Reduce training data size
num_samples = 30000  # Adjust based on available memory and desired speed
x_train_flat, y_train = x_train_flat[:num_samples], y_train[:num_samples]
x_train, y_train_images = x_train[:num_samples], y_train[:num_samples]

data_loading_time = time.time() - start_time
print(f"Data loading and preprocessing time: {data_loading_time:.2f} seconds")

# Step 2: Standardize Data
start_time = time.time()
scaler = StandardScaler()
x_train_flat = scaler.fit_transform(x_train_flat)
x_test_flat = scaler.transform(x_test_flat)

pca = PCA(n_components=0.90)  # Retain 90% of the variance
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)
preprocessing_time = time.time() - start_time
print(f"Data scaling and PCA time: {preprocessing_time:.2f} seconds")

# Step 3: Hyperparameter Search
start_time = time.time()
param_dist = {'C': [1.8, 1.9, 2, 2.1, 2.2], 'gamma': [0.01, 0.1, 1, 'scale']}
random_search = RandomizedSearchCV(SVC(kernel='rbf'), param_dist, n_iter=5, cv=3, n_jobs=-1, verbose=2)
random_search.fit(x_train_pca, y_train)

best_params = random_search.best_params_
hyperparam_search_time = time.time() - start_time
print(f"Hyperparameter search time: {hyperparam_search_time:.2f} seconds")
print("Best Hyperparameters:", best_params)

# Step 4: Train SVM
start_time = time.time()
svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
svm.fit(x_train_pca, y_train)

# Step 5: Evaluate Performance
start_time = time.time()
y_pred_train = svm.predict(x_train_pca)
y_pred_test = svm.predict(x_test_pca)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

evaluation_time = time.time() - start_time
print(f"Model evaluation time: {evaluation_time:.2f} seconds")

# Print Results
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 6: Visualize Correct and Incorrect Classifications
def display_examples(x_images, y_true, y_pred, classes, num_examples=5):
    """Displays examples of correct and incorrect classifications."""
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]

    # Display Correct Classifications
    print("\nExamples of Correct Classifications:")
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(np.random.choice(correct_indices, num_examples, replace=False)):
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(x_images[idx])
        plt.title(f"True: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}")
        plt.axis('off')

    # Display Incorrect Classifications
    print("\nExamples of Incorrect Classifications:")
    for i, idx in enumerate(np.random.choice(incorrect_indices, num_examples, replace=False)):
        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(x_images[idx])
        plt.title(f"True: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}")
        plt.axis('off')

    plt.show()

# Map numeric labels to class names
class_names = {3: "Cat", 5: "Dog"}
y_test_original = np.where(y_test == -1, 3, 5)
y_pred_original = np.where(y_pred_test == -1, 3, 5)

# Display examples
display_examples(x_test, y_test_original, y_pred_original, class_names)

# Total Execution Time
total_execution_time = time.time() - total_start_time
print(f"Total execution time: {total_execution_time:.2f} seconds")
