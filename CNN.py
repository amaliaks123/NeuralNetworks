import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


# early stopping, dropout, learning rate scheduler, GlobalAveragePooling2D
# batch 64


#load data
(xtrain, ytrain), (xtest, ytest) = cifar100.load_data()

#regularization of data
xtrain = xtrain.astype('float32') / 255.0
xtest = xtest.astype('float32') / 255.0

#Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=5,         # Περιστροφή έως 10 μοίρες
    width_shift_range=0.05,     # Οριζόντια μετατόπιση έως 10%
    height_shift_range=0.05,    # Κάθετη μετατόπιση έως 10%
    horizontal_flip=True       # Κατοπτρισμός
)
datagen.fit(xtrain)

#one-hot encoding
ytrain = to_categorical(ytrain, 100)
ytest = to_categorical(ytest, 100)

# Learning rate exponential decay function
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=50000,  # Number of steps to decay over
    decay_rate=0.98,     # Rate of decay
    staircase=True       # If True, decay is applied at discrete intervals
)

#create cnn
model = Sequential([
    #input shape explicitly
    Input(shape=(32, 32, 3)),

    # First convolutional block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Second convolutional block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    # Third convolutional block
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    # Fourth convolutional block
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # fifth convolutional block
    Conv2D(1024, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # GlobalAveragePooling2D and Dense layers
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(100, activation='softmax')
])


#compile model & small learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',       # Monitors validation loss
    patience=5,               # Stops after 5 epochs of no improvement
    restore_best_weights=True # Restores the best weights for evaluation
)

#train model
history = model.fit(datagen.flow(xtrain, ytrain, batch_size=64),
                    epochs=25,
                    validation_data=(xtest, ytest),
                    verbose=1,
                    callbacks=[early_stopping])

#evaluate model
test_loss, test_acc = model.evaluate(xtest, ytest)
print('Test accuracy:', test_acc)



#accuracy curve
plt.plot(history.history['accuracy'], label='Ακρίβεια Εκπαίδευσης')
plt.plot(history.history['val_accuracy'], label='Ακρίβεια Ελέγχου')
plt.xlabel('Εποχές')
plt.ylabel('Ακρίβεια')
plt.title('Ακρίβεια Εκπαίδευσης και Ελέγχου ανά Εποχή')
plt.legend()
plt.show()

#loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.title('Απώλεια Εκπαίδευσης & Testing')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#example of correct and wrong classification
# predictions
ypred = np.argmax(model.predict(xtest), axis=1)
ytrue = np.argmax(ytest, axis=1)

#correctly classifies images
correct = np.where(ypred == ytrue)[0]

#wrongly classified images
incorrect = np.where(ypred != ytrue)[0]


#names of classes
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
    'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
    'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree',
    'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
    'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm'
   ]

#show examples
def plot_examples(indices, title):
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(xtest[idx])
        plt.title(f"Πραγματικό: {class_names[ytrue[idx]]}\nΠρόβλεψη: {class_names[ypred[idx]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Εμφάνιση σωστών παραδειγμάτων
plot_examples(correct, "Σωστές Κατηγοριοποιήσεις")

# Εμφάνιση λανθασμένων παραδειγμάτων
plot_examples(incorrect, "Λανθασμένες Κατηγοριοποιήσεις")

