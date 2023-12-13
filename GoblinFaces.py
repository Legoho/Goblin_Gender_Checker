import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap


notebookdir=os.getcwd()
folder_path=r"archive\images\gob64"
file_path = 'archive\artbreederGoblin.csv'

file_names = os.listdir(folder_path)
print(f"Number of filenames: {len(file_names)}")
id_column="idArtBreeder"
label_column = 'gender'
df = pd.read_csv(file_path)
ids=df[id_column].apply(lambda x: os.path.join(folder_path, x + '-64.jpg'))
# Assuming you have a list of file paths and corresponding labels
# Get a list of all files in the image folder



print(f"Number of filenames: {len(file_names)}")


# List of file paths to your images
labels =  df[label_column] # List of corresponding labels (1 for male, 5 for female others in between)

print(f"Number of Labels: {len(labels)}")
labels = to_categorical(labels-1, num_classes=5)


# Shuffle the data
ids, labels = shuffle(ids, labels, random_state=42)

# Split the dataset into training and testing sets
train_files, test_files, train_labels, test_labels = train_test_split(
    ids, labels, test_size=0.2, random_state=42
)
# Data preprocessing function
def preprocess_image(file_path, label):
    # Load and preprocess the image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    #img = tf.image.resize(img, (224, 224))  # Adjust the size as needed
    img = tf.cast(img, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return img, label

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(preprocess_image)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size=32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_dataset = test_dataset.map(preprocess_image)
test_dataset = test_dataset.batch(batch_size=32)


# Define the model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dense(128, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))



# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model. summary()

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# predict probabilities for test set
predictions_raw = model.predict(test_dataset)
# predict crisp classes for test set
predicted_classes = np.argmax(predictions_raw, axis=1)
# reduce to 1d array
predictions_raw
print(predicted_classes)
print(predictions_raw)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')

test_labels = np.argmax(test_labels+1, axis=1)

print(test_labels)
print(predicted_classes)

precision = precision_score(test_labels, predicted_classes, average='weighted')
recall = recall_score(test_labels, predicted_classes, average='weighted')
f1 = f1_score(test_labels, predicted_classes, average='weighted')

print(precision)
print(recall)
print(f1)

kappa = cohen_kappa_score(test_labels, predicted_classes)
print(f'Cohen\'s Kappa: {kappa}')

cm = confusion_matrix(test_labels, predicted_classes)
print(f'Confusion Matrix:\n{cm}')

y_true_binarized = label_binarize(test_labels, classes=np.unique(test_labels))

roc_auc_scores = []

for i in range(np.max(test_labels) + 1):  # Assuming class labels start from 0
    roc_auc_scores.append(roc_auc_score(y_true_binarized[:, i], predictions_raw[:, i]))

print(f'ROC AUC Scores for Each Class: {roc_auc_scores}')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
