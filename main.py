from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

import warnings
warnings.filterwarnings("ignore")

# Base Path for all files
base_dir = r"C:\Users\prera\OneDrive\Desktop\Imarticus\ML\datasets\capstone2\2\train"

# Initialize an empty dictionary to store class counts
class_counts = {}

# Loop through the subdirectories (classes) in your dataset directory
for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if os.path.isdir(class_dir):
        # Count the number of images in each class
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images

# Extract class names and counts
class_names = list(class_counts.keys())
class_counts = list(class_counts.values())

# Create a color map for the bars
colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))

# Create a bar plot to visualize class distribution
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, class_counts, color=colors)
plt.xlabel('Class Names', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.title('Class Distribution of Image Dataset', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Add text labels with the exact counts above each bar
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
             ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()

num_samples = 5 

# Create a list of class names (subdirectories)
class_names = os.listdir(base_dir)

# Initialize a Matplotlib figure for displaying the sample images
plt.figure(figsize=(15, 5))

# Loop through the class names to display random samples
for class_name in class_names:
    class_dir = os.path.join(base_dir, class_name)
    if os.path.isdir(class_dir):
        # List all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]

        # Randomly select and display a sample of images
        for i in range(num_samples):
            # Randomly select an image file
            random_image_file = random.choice(image_files)

            # Load and display the image using OpenCV
            image_path = os.path.join(class_dir, random_image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            plt.subplot(len(class_names), num_samples, len(class_names) * i + class_names.index(class_name) + 1)
            plt.imshow(image)
            plt.title(class_name)
            plt.axis('off')

# Adjust spacing between subplots and display the figure
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

datagenerator = {
    "train": ImageDataGenerator(horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1. / 255,
                                validation_split=0.1,
                                shear_range=0.1,
                                zoom_range=0.1,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=30,
                               ).flow_from_directory(directory=base_dir,
                                                     target_size=(300, 300),
                                                     subset='training',
                                                    ),

    "valid": ImageDataGenerator(rescale=1. / 255,
                                validation_split=0.1,
                               ).flow_from_directory(directory=base_dir,
                                                     target_size=(300, 300),
                                                     subset='validation',
                                                    ),
}

# Create a generator for images in the specified subset
image_generator = datagenerator["train"]

# Generate and visualize augmented images
num_samples = 5  # Adjust the number of augmented images to display
plt.figure(figsize=(12, 6))

for i in range(num_samples):
    augmented_image, class_label = next(image_generator)  # Generate the next augmented image
    image = augmented_image[0]  # Extract the image data from the generator output
    class_index = np.argmax(class_label[0])  # Convert one-hot encoded label to index

    # Display the augmented image
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(image)
    plt.title(f'Class Index: {class_index}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Initializing InceptionV3 (pretrained) model with input image shape as (300, 300, 3)
base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3)) #fully connected layers (top layers) will not be included

# Load Weights for the InceptionV3 Model
base_model.load_weights(r"C:\Users\prera\OneDrive\Desktop\Imarticus\ML\datasets\capstone2\2\abc\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Setting the Training of all layers of InceptionV3 model to false
base_model.trainable = False #freezing all the layers of the InceptionV3 model

# Adding some more layers at the end of the Model as per our requirement
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(1024, activation='relu'),
    Dense(3, activation='softmax') 
])

# Using the Adam Optimizer to set the learning rate of our final model
o = optimizers.Adam(learning_rate=0.0001)

# Compiling and setting the parameters we want our model to use
model.compile(loss="categorical_crossentropy", optimizer= o, metrics=['accuracy'])

# Setting variables for the model
batch_size = 32
epochs = 10

# Seperating Training and Testing Data
train_generator = datagenerator["train"]
valid_generator = datagenerator["valid"]

# Calculating variables for the model
steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

print("steps_per_epoch :", steps_per_epoch)
print("validation_steps :", validation_steps)

# File Path to store the trained models
filepath = "./model_{epoch:02d}-{val_accuracy:.2f}.h5"

# Using the ModelCheckpoint function to train and store all the best models
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint1]
# Training the Model
history = model.fit_generator(generator=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=valid_generator, validation_steps=validation_steps,
                              callbacks=callbacks_list)

# Check our folder and import the model with best validation accuracy
loaded_best_model = keras.models.load_model("./model_08-0.81.h5")

# Custom function to predict label for the image
def predict(img_path):
    # Load the image from file path
    img = image.load_img(img_path, target_size=(300, 300))
    
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    
    # Scaling the Image Array values between 0 and 1
    img = img / 255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    # Get the Predicted Label for the loaded Image
    p = loaded_best_model.predict(img[np.newaxis, ...])

    # Label array
    labels = {0: 'adidas', 1: 'converse', 2: 'nike'}

    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")

    classes = []
    prob = []
    print("\n-------------------Individual Probability--------------------------------\n")

    for i, j in enumerate(p[0], 0):
        print(labels[i].upper(), ':', round(j * 100, 2), '%')
        classes.append(labels[i])
        prob.append(round(j * 100, 2))

    def plot_bar_x():
        # this is for plotting purpose
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=8)
        plt.ylabel('Probability', fontsize=8)
        plt.xticks(index, classes, fontsize=8, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()

    plot_bar_x()


