import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


#In this part, the code reads a file named 'labels_new.txt' that contains mappings between numerical labels and corresponding glosses. 
#It creates a dictionary (label_to_gloss) to store these mappings.
label_to_gloss = {}
with open('Labels.txt', 'r') as file:
    for line in file:
        label, gloss = line.strip().split()
        label_to_gloss[int(label)] = gloss

print(label_to_gloss)

#The dataset is already pre-preocessed and stored which is being called here
preprocessed_data_dir = 'C:\\Users\\rk_ja\\Desktop\\Pre_trained_model\\DATASET4'

# Get the list of class (label) directories
class_directories = [f.path for f in os.scandir(preprocessed_data_dir) if f.is_dir()]

#New directories are created for the training and the testing sets
train_dir = os.path.join(preprocessed_data_dir, 'train')
test_dir = os.path.join(preprocessed_data_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#Iterating through each class directory and extracting the label from the directory name, with that finding the corresponding gloss. 
#Then spliting the dataset into training and testing sets using train_test_split.
#Moving the images to their respective directories.

for class_dir in class_directories:
    try:
        label = int(os.path.basename(class_dir))
    except ValueError:
        # Skipping directories that cannot be converted to an integer (e.g., 'train' or 'test')
        continue
    
    gloss = label_to_gloss[label]
    
    class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.jpg')]
    
    # Split the dataset into training and testing sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        class_files, [label] * len(class_files), test_size=0.2, random_state=42, stratify=[label] * len(class_files)
    )
    
    # Moving images to their respective directories
    for path in train_paths:
        dest_path = os.path.join(train_dir, str(label), os.path.basename(path))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(path, dest_path)

    for path in test_paths:
        dest_path = os.path.join(test_dir, str(label), os.path.basename(path))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(path, dest_path)

# Defining the image size and batch size
img_size = (224, 224)
batch_size = 32

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

#way to efficiently load and preprocess data in batches during training
# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',  
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',  
    shuffle=False  
)

# Loading the  pre-trained MobileNetV2 model
train_labels = []
for i in range(train_generator.samples // batch_size):
    _, labels = train_generator.next()
    train_labels.extend(labels.astype(int))


# Manually compute class weights
unique_classes, class_counts = np.unique(train_labels, return_counts=True)
total_samples = len(train_labels)
class_weights = total_samples / (len(unique_classes) * class_counts)
class_weight_dict = dict(zip(unique_classes, class_weights))

# Manually compute class weights
#class_weight_dict = {
#    class_label: weight
#    for class_label, weight in zip(np.unique(train_labels), [1.0, 1.0, 1.5, 1.5, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # Adjust weights manually
#}


# Loading the  pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freezing the layers of the pre-trained model so that the weights during training are retained
for layer in base_model.layers:
    layer.trainable = False

# Flattening the output of the base model
flattened_features = layers.Flatten()(base_model.output)

# Dense layer is added for classification with softmax function
output_layer = layers.Dense(len(label_to_gloss), activation='softmax')(flattened_features)

#A new Global Average Pooling layer and a Dense layer with softmax activation are added to adapt the model for the specific task.
#Global average pooling reduces the spatial dimensions of the feature maps to a single value per channel, 
#resulting in a compact representation that is more computationally efficient
#Instead of fully connected layers, MobileNetV2 uses global average pooling at the end of the network.
#model = models.Sequential([
#    base_model,
#    layers.GlobalAveragePooling2D(),
#    layers.Dense(len(label_to_gloss), activation='softmax')  #The length of label_to_gloss dictionary is given in the final layer
#])


# Creating the model
model = models.Model(inputs=base_model.input, outputs=output_layer)

#The optimizer is responsible for adjusting the weights of the neural network during training to minimize the error
#The loss function (or objective function) is a measure of how well the model's predictions match the actual target values. 
#It quantifies the difference between the predicted output and the ground truth.
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    class_weight=class_weight_dict
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc}')

# After model evaluation
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)

# Save the model
model.save('sign_language_nlp_model_without_attention.h5')
