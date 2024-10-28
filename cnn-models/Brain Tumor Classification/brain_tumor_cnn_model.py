# Import libraries for data handling, visualization, image processing, and augmentation
import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.src.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import imutils
from sklearn.utils import shuffle

# Set plot style for consistent visualization
plt.style.use('ggplot')

# Dataset folder paths for tumor and non-tumor images
dataset_folderY = 'dataset_brain_tumor/brain_tumor_yes/'
dataset_folderN = 'dataset_brain_tumor/brain_tumor_no/'
dataset_folderA = 'dataset_brain_tumor/augmented_brain_tumor_images/'

# Ensure output directories for augmented data exist
os.makedirs(dataset_folderA + 'brain_tumor_yes', exist_ok=True)
os.makedirs(dataset_folderA + 'brain_tumor_no', exist_ok=True)


# Function to rename images with a specific prefix and avoid file name conflicts
def rename_images(folder_path, prefix):
    """Rename images in a folder with a specified prefix to avoid name conflicts."""
    image_count = 1
    for image_file in os.listdir(folder_path):
        source = os.path.join(folder_path, image_file)
        destination = os.path.join(folder_path, f"{prefix}_{image_count}.jpg")

        # Check for existing files and increment image_count if needed
        while os.path.exists(destination):
            image_count += 1
            destination = os.path.join(folder_path, f"{prefix}_{image_count}.jpg")

        try:
            os.rename(source, destination)
        except Exception as e:
            print(f"Error renaming {source}: {e}")
        image_count += 1


# Rename images in each folder to ensure consistent naming
rename_images(dataset_folderY, "Brain_Tumor_Yes")
rename_images(dataset_folderN, "Brain_Tumor_No")
print("All images have been renamed successfully.")


# Count images in each category folder for data analysis
def count_images(folder_path):
    """Return the number of image files in a folder."""
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])


count_image_files_yes = count_images(dataset_folderY)
count_image_files_no = count_images(dataset_folderN)
print(f"Tumor Data: {count_image_files_yes}")
print(f"Non-Tumor Data: {count_image_files_no}")


# Function to visualize data distribution using a bar chart
def plot_image_counts(counts):
    """Plot a bar chart of image counts for each category."""
    plt.figure(figsize=(6, 9))
    plt.bar(counts.keys(), counts.values(), color="blue")
    plt.xlabel("Total Data per Classification")
    plt.ylabel("No. of Images")
    plt.title("Count of Brain Tumor/Non Brain Tumor Images")
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a grid
    plt.show()


# Visualize the data distribution
plot_image_counts({"Tumor Data": count_image_files_yes, "Non Tumor Data": count_image_files_no})


# Utility function for formatting elapsed time in hours, minutes, seconds
def timing(time_elapsed):
    """Format elapsed time as hours:minutes:seconds."""
    hour = int(time_elapsed / (60 * 60))
    minutes = int(time_elapsed % (60 * 60) / 60)
    seconds = int(time_elapsed % 60)
    return f"{hour}:{minutes}:{seconds}"


# Function for data augmentation to expand dataset with transformed images
def augmented_data(file_dir, n_generated_samples, save_to_dir):
    """Perform data augmentation and save augmented images to a directory."""
    # Check if the directory already has augmented images
    if len(os.listdir(save_to_dir)) > 0:
        print(f"Augmented images already exist in {save_to_dir}. Skipping augmentation.")
        return

    # Define augmentation transformations
    augmented_image_data = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    for image_file in os.listdir(file_dir):
        image_path = os.path.join(file_dir, image_file)

        try:
            # Read and reshape image for augmentation
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                continue

            image = image.reshape((1,) + image.shape)
            save_image_prefix = f"aug_{os.path.splitext(image_file)[0]}"
            i = 0

            # Generate augmented images and save to directory
            for batch in augmented_image_data.flow(
                    x=image,
                    batch_size=1,
                    save_to_dir=save_to_dir,
                    save_prefix=save_image_prefix,
                    save_format="jpg"
            ):
                i += 1
                if i >= n_generated_samples:
                    break
        except Exception as e:
            print(f"Error augmenting {image_file}: {e}")


# Measure time taken for data augmentation and perform augmentation
start_time = time.time()

augmented_data(file_dir=dataset_folderY, n_generated_samples=12,
               save_to_dir=os.path.join(dataset_folderA, 'brain_tumor_yes'))
augmented_data(file_dir=dataset_folderN, n_generated_samples=18,
               save_to_dir=os.path.join(dataset_folderA, 'brain_tumor_no'))

end_time = time.time()
processing_time = end_time - start_time
print("Augmentation completed in:", timing(processing_time))


# Function to summarize the image data after augmentation
def image_data_summary():
    """Print summary of augmented image data."""
    dataset_folderAY = 'dataset_brain_tumor/augmented_brain_tumor_images/brain_tumor_yes/'
    dataset_folderAN = 'dataset_brain_tumor/augmented_brain_tumor_images/brain_tumor_no/'

    image_aug_no = len(os.listdir(dataset_folderAN))
    image_aug_yes = len(os.listdir(dataset_folderAY))

    total_aug_image = (image_aug_no + image_aug_yes)

    image_percentage_no = (image_aug_no * 100) / total_aug_image if total_aug_image > 0 else 0
    image_percentage_yes = (image_aug_yes * 100) / total_aug_image if total_aug_image > 0 else 0

    print(f"Number of Sample: {total_aug_image}")
    print(f"{image_aug_yes} Number of positive samples in percentage: {image_percentage_yes}%")
    print(f"{image_aug_no} Number of negative samples in percentage: {image_percentage_no}%")


# Call the summary function to display augmented data details
image_data_summary()

# Count the augmented images for final verification
count_image_files_yes = count_images(dataset_folderA + "brain_tumor_yes/")
count_image_files_no = count_images(dataset_folderA + "brain_tumor_no/")
print(f"Augmented Tumor Data: {count_image_files_yes}")
print(f"Augmented Non-Tumor Data: {count_image_files_no}")

# Function to visualize augmented data distribution using a bar chart
plot_image_counts({"Augmented Tumor Data": count_image_files_yes, "Augmented Non Tumor Data": count_image_files_no})


# Data Preprocessing: Crop the images to focus on the brain tumor
def crop_brain_tumor_img(image, plot=False):
    """Crop the brain tumor region from the image and optionally plot results."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    image_threshold = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)[1]
    image_threshold = cv2.erode(image_threshold, None, iterations=3)
    image_threshold = cv2.dilate(image_threshold, None, iterations=3)

    image_contours = cv2.findContours(image_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = imutils.grab_contours(image_contours)

    max_contours_image = max(image_contours, key=cv2.contourArea)

    ext_left = tuple(max_contours_image[max_contours_image[:, :, 0].argmin()][0])
    ext_right = tuple(max_contours_image[max_contours_image[:, :, 0].argmax()][0])
    ext_top = tuple(max_contours_image[max_contours_image[:, :, 1].argmin()][0])
    ext_bottom = tuple(max_contours_image[max_contours_image[:, :, 1].argmax()][0])

    image_after_contours = image[ext_top[1]:ext_bottom[1], ext_left[0]:ext_right[0]]

    if plot:
        # Plotting the original and cropped images for comparison with grid
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.tick_params(axis="both", which="both", top=False, bottom=False, left=False, right=False, labelbottom=False,
                        labeltop=False, labelleft=False, labelright=False)

        plt.subplot(1, 2, 2)
        plt.imshow(image_after_contours)
        plt.title('Cropped Image')

        plt.tick_params(axis="both", which="both", top=False, bottom=False, left=False, right=False, labelbottom=False,
                        labeltop=False, labelleft=False, labelright=False)

        plt.show()

    return image_after_contours


# Example usage of the augmented data function
dataset_folderAY = 'dataset_brain_tumor/augmented_brain_tumor_images/brain_tumor_yes/'
dataset_folderAN = 'dataset_brain_tumor/augmented_brain_tumor_images/brain_tumor_no/'

for image_file in os.listdir(dataset_folderAY):
    image_data = cv2.imread(dataset_folderAY + image_file)
    image_data = crop_brain_tumor_img(image_data, False)
    cv2.imwrite(dataset_folderAY + image_file, image_data)


for image_file in os.listdir(dataset_folderAN):
    image_data = cv2.imread(dataset_folderAN + image_file)
    image_data = crop_brain_tumor_img(image_data, False)
    cv2.imwrite(dataset_folderAN + image_file, image_data)

augmented_data(dataset_folderY, n_generated_samples=12, save_to_dir=dataset_folderAY)
augmented_data(dataset_folderN, n_generated_samples=18, save_to_dir=dataset_folderAN)


def load_image_data(dir_list, image_size):
    x = []
    y = []

    image_width, image_height = image_size

    for augmented_image_data in dir_list:
        for filename in os.listdir(augmented_image_data):
            test_image = cv2.imread(augmented_image_data + "/" + filename)
            cropped_image = crop_brain_tumor_img(test_image, plot=True)
            resize_image = cv2.resize(cropped_image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

            image = resize_image / 255.00

            x.append(image)

            if augmented_image_data[-3:] == "yes":
                y.append(1)
            else:
                y.append(0)

    x = np.array(x)
    y = np.array(y)

    x, y = shuffle(x, y)
    print(f"Number of example is : {len((x))}")
    print(f"x SHAPE is : {x.shape}")
    print(f"y SHAPE is : {y.shape}")

    return x, y


dataset_folderAY = 'dataset_brain_tumor/augmented_brain_tumor_images/brain_tumor_yes'
dataset_folderAN = 'dataset_brain_tumor/augmented_brain_tumor_images/brain_tumor_no'

IMAGE_WIDTH, IMAGE_HEIGHT = (240, 120)

# x,y = load_image_data([dataset_folderAY, dataset_folderAN], (IMAGE_WIDTH, IMAGE_HEIGHT))

#Data Spliting
base_dir = 'brain_image_data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

if not os.path.isdir('brain_image_data'):
    os.mkdir(base_dir)

#Tranning Data
if not os.path.isdir('brain_image_data/train'):
    os.mkdir(train_dir)

#Testing Data
if not os.path.isdir('brain_image_data/test'):
    os.mkdir(test_dir)

#Data Spliting
if not os.path.isdir('brain_image_data/valid'):
    os.mkdir(valid_dir)

if not os.path.isdir('brain_image_data/train/brain_tumor_yes'):
    infected_train_image = os.path.join(train_dir, 'brain_tumor_yes')
    os.mkdir(infected_train_image)
if not os.path.isdir('brain_image_data/test/brain_tumor_yes'):
    infected_test_image = os.path.join(test_dir, 'brain_tumor_yes')
    os.mkdir(infected_test_image)
if not os.path.isdir('brain_image_data/valid/brain_tumor_yes'):
    infected_valid_image = os.path.join(valid_dir, 'brain_tumor_yes')
    os.mkdir(infected_valid_image)

if not os.path.isdir('brain_image_data/train/brain_tumor_no'):
    train_image = os.path.join(train_dir, 'brain_tumor_no')
    os.mkdir(train_image)
if not os.path.isdir('brain_image_data/test/brain_tumor_no'):
    test_image = os.path.join(test_dir, 'brain_tumor_no')
    os.mkdir(test_image)
if not os.path.isdir('brain_image_data/valid/brain_tumor_no'):
    valid_image = os.path.join(valid_dir, 'brain_tumor_no')
    os.mkdir(valid_image)

files = os.listdir(dataset_folderAY)
image_file_name = []
for augmented_image in range(0, 1301):
    image_file_name.append(files[augmented_image])
for image in image_file_name:
    image_source = os.path.join(dataset_folderAY, image)
    image_destination = os.path.join(train_dir, 'brain_tumor_yes', image)
    shutil.copyfile(image_source, image_destination)

files = os.listdir(dataset_folderAY)
image_file_name = []
for augmented_image in range(1301, 1673):
    image_file_name.append(files[augmented_image])
for image in image_file_name:
    image_source = os.path.join(dataset_folderAY, image)
    image_destination = os.path.join(test_dir, 'brain_tumor_yes', image)
    shutil.copyfile(image_source, image_destination)

files = os.listdir(dataset_folderAY)
image_file_name = []
for augmented_image in range(1673, 1858):
    image_file_name.append(files[augmented_image])
for image in image_file_name:
    image_source = os.path.join(dataset_folderAY, image)
    image_destination = os.path.join(valid_dir, 'brain_tumor_yes', image)
    shutil.copyfile(image_source, image_destination)

files = os.listdir(dataset_folderAN)
image_file_name = []
for augmented_image in range(0, 1232):
    image_file_name.append(files[augmented_image])
for image in image_file_name:
    image_source = os.path.join(dataset_folderAN, image)
    image_destination = os.path.join(train_dir, 'brain_tumor_no', image)
    shutil.copyfile(image_source, image_destination)

files = os.listdir(dataset_folderAN)
image_file_name = []
for augmented_image in range(1232, 1585):
    image_file_name.append(files[augmented_image])
for image in image_file_name:
    image_source = os.path.join(dataset_folderAN, image)
    image_destination = os.path.join(test_dir, 'brain_tumor_no', image)
    shutil.copyfile(image_source, image_destination)

files = os.listdir(dataset_folderAN)
image_file_name = []
for augmented_image in range(1585, 1761):
    image_file_name.append(files[augmented_image])
for image in image_file_name:
    image_source = os.path.join(dataset_folderAN, image)
    image_destination = os.path.join(valid_dir, 'brain_tumor_no', image)
    shutil.copyfile(image_source, image_destination)

#Building the Model
train_image_data = ImageDataGenerator(rescale=1.0 / 255, rotation_range=40, width_shift_range=0.4,
                                      height_shift_range=0.4, shear_range=0.2, brightness_range=(0.3, 1.0),
                                      horizontal_flip=0.4, vertical_flip=0.4, fill_mode="nearest")

test_image_data = ImageDataGenerator(rescale=1.0 / 255)

valid_image_data = ImageDataGenerator(rescale=1.0 / 255)

train_image_generator = train_image_data.flow_from_directory('brain_image_data/train/', batch_size=32,
                                                             target_size=(240, 240), class_mode='categorical',
                                                             shuffle=True, seed=42, color_mode='rgb')

test_image_generator = test_image_data.flow_from_directory('brain_image_data/train/', batch_size=32,
                                                           target_size=(240, 240), class_mode='categorical',
                                                           shuffle=True, seed=42, color_mode='rgb')

valid_image_generator = valid_image_data.flow_from_directory('brain_image_data/train/', batch_size=32,
                                                             target_size=(240, 240), class_mode='categorical',
                                                             shuffle=True, seed=42, color_mode='rgb')

image_class_label = train_image_generator.class_indices

image_class_name = {value: key for (key, value) in image_class_label.items()}

base_model = VGG19(input_shape = (240,240,3), include_top=False, weights='imagenet')

for layers in base_model.layers:
    layers.trainable = False


image_base_model=base_model.output
flat = Flatten()(image_base_model)

first_layer_class = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.3)(first_layer_class)
second_layer_class = Dense(1152, activation = 'relu')(drop_out)
output_layer = Dense(2, activation = 'softmax')(first_layer_class)

image_model = Model(base_model.input, output_layer)

es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=4)
cp = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
                     mode='auto', save_freq='epoch')
lrr = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, patience=3, factor=0.5, min_lr=0.0001)

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,  # adjust this based on dataset size and requirements
    decay_rate=0.96,    # adjust decay rate as needed
    staircase=False     # set to True if you want a step-wise decay
)
# Initialize SGD optimizer with the learning rate schedule
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

image_model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics =['accuracy'])
image_model.summary()

training_model = image_model.fit(train_image_generator, steps_per_epoch = 100, epochs = 50, callbacks=[es, cp, lrr],
                                 validation_data = valid_image_generator)

# training_model = image_model.fit(train_image_generator, steps_per_epoch = 10, epochs = 5, callbacks=[es, cp, lrr],
#                                  validation_data = valid_image_generator)

# Plotting the accuracy and loss
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
fig.suptitle("Model Training - CNN", fontsize=14)
max_epoch = len(training_model.history['accuracy']) + 1
epoch_list = list(range(1, max_epoch))

ax1.plot(epoch_list, training_model.history['accuracy'], color='green', linestyle='-', label='Training Data')
ax1.plot(epoch_list, training_model.history['val_accuracy'], color='red', linestyle='-', label='Validation Data')
ax1.set_title("Training Accuracy", fontsize=12)
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.legend(frameon=False, loc='lower center', ncol=2)

ax2.plot(epoch_list, training_model.history['loss'], color='green', linestyle='-', label='Training Data')
ax2.plot(epoch_list, training_model.history['val_loss'], color='red', linestyle='-', label='Validation Data')
ax2.set_title("Training Loss", fontsize=12)
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.legend(frameon=False, loc='lower center', ncol=2)

plt.savefig("training_cnn_line_plot.jpg", format='jpg', dpi=100, bbox_inches='tight')

if not os.path.isdir('model_weights/'):
    os.mkdir("model_weights/")
image_model.save_weights(filepath="model_weights/vgg19_image_model.weights.h5", overwrite=True)

image_model.load_weights("model_weights/vgg19_image_model.weights.h5")
valid_image_generator_evaluation = image_model.evaluate(valid_image_generator)
test_image_generator_evaluation = image_model.evaluate(test_image_generator)

print(f'Validation Loss: {valid_image_generator_evaluation[0]}')
print(f'Validation Accuracy: {valid_image_generator_evaluation[1]}')
print(f'Test Loss: {test_image_generator_evaluation[0]}')
print(f'Test Accuracy: {test_image_generator_evaluation[1]}')

image_file_name = test_image_generator.filenames
no_of_sample = len(image_file_name)

valid_image_generator_prediction = image_model.predict(test_image_generator, steps=no_of_sample, verbose=1)
y_prediction = np.argmax(valid_image_generator_prediction, axis=1)

#Incremental and fine tuning
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
base_model_layer_names = [layer.name for layer in base_model.layers]

image_base_model=base_model.output
flat = Flatten()(image_base_model)

first_layer_class = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.3)(first_layer_class)
second_layer_class = Dense(1152, activation = 'relu')(drop_out)
output_layer = Dense(2, activation = 'softmax')(first_layer_class)

image_model_02 = Model(base_model.inputs, output_layer)
image_model_02.load_weights('model_weights/vgg19_image_model.weights.h5')

set_trainable_image = False
for layer in base_model.layers:
    if layer.name in ['block5_conv3', 'block5_conv4']:
        set_trainable_image = True
    if set_trainable_image:
        layer.trainable=True
    else:
        layer.trainable=False


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,  # adjust this based on dataset size and requirements
    decay_rate=0.96,    # adjust decay rate as needed
    staircase=False     # set to True if you want a step-wise decay
)
# Initialize SGD optimizer with the learning rate schedule
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

image_model_02.compile(loss='categorical_crossentropy', optimizer = sgd, metrics =['accuracy'])
print(image_model_02.summary())

training_model_02 = image_model_02.fit(train_image_generator, steps_per_epoch = 100, epochs = 50,
                                       callbacks=[es, cp, lrr], validation_data = valid_image_generator)

# training_model_02 = image_model_02.fit(train_image_generator, steps_per_epoch = 10, epochs = 5, callbacks=[es, cp, lrr],
#                                        validation_data = valid_image_generator)

# Plotting the accuracy and loss
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
fig.suptitle("Model Training - CNN", fontsize=14)
max_epoch = len(training_model_02.history['accuracy']) + 1
epoch_list = list(range(1, max_epoch))

ax1.plot(epoch_list, training_model_02.history['accuracy'], color='green', linestyle='-', label='Training Data')
ax1.plot(epoch_list, training_model_02.history['val_accuracy'], color='red', linestyle='-', label='Validation Data')
ax1.set_title("Training Accuracy", fontsize=12)
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.legend(frameon=False, loc='lower center', ncol=2)

ax2.plot(epoch_list, training_model_02.history['loss'], color='green', linestyle='-', label='Training Data')
ax2.plot(epoch_list, training_model_02.history['val_loss'], color='red', linestyle='-', label='Validation Data')
ax2.set_title("Training Loss", fontsize=12)
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.legend(frameon=False, loc='lower center', ncol=2)

plt.savefig("training_cnn_line_plot_02.jpg", format='jpg', dpi=100, bbox_inches='tight')

if not os.path.isdir('model_weights/'):
    os.mkdir("model_weights/")
image_model_02.save_weights(filepath="model_weights/vgg19_image_model_02.weights.h5", overwrite=True)

image_model_02.load_weights("model_weights/vgg19_image_model_02.weights.h5")
valid_image_generator_evaluation_02 = image_model_02.evaluate(valid_image_generator)
test_image_generator_evaluation_02 = image_model_02.evaluate(test_image_generator)

#unfreazing
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
base_model_layer_names = [layer.name for layer in base_model.layers]

image_base_model=base_model.output
flat = Flatten()(image_base_model)

first_layer_class = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.3)(first_layer_class)
second_layer_class = Dense(1152, activation = 'relu')(drop_out)
output_layer = Dense(2, activation = 'softmax')(first_layer_class)

image_model_03 = Model(base_model.inputs, output_layer)
image_model_03.load_weights('model_weights/vgg19_image_model_02.weights.h5')

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,  # adjust this based on dataset size and requirements
    decay_rate=0.96,    # adjust decay rate as needed
    staircase=False     # set to True if you want a step-wise decay
)
# Initialize SGD optimizer with the learning rate schedule
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

image_model_03.compile(loss='categorical_crossentropy', optimizer = sgd, metrics =['accuracy'])
print(image_model_03.summary())

training_model_03 = image_model_03.fit(train_image_generator, steps_per_epoch = 100, epochs = 50,
                                       callbacks=[es, cp, lrr], validation_data = valid_image_generator)

# training_model_03 = image_model_03.fit(train_image_generator, steps_per_epoch = 10, epochs = 5, callbacks=[es, cp, lrr],
#                                        validation_data = valid_image_generator)

if not os.path.isdir('model_weights/'):
    os.mkdir("model_weights/")
image_model_03.save_weights(filepath="model_weights/vgg19_image_model_03.weights.h5", overwrite=True)

image_model_03.load_weights("model_weights/vgg19_image_model_03.weights.h5")
valid_image_generator_evaluation_03 = image_model_03.evaluate(valid_image_generator)
test_image_generator_evaluation_03 = image_model_03.evaluate(test_image_generator)