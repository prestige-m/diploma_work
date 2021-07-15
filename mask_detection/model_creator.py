import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

from numpy import ndarray
import os

# Params
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
IMG_DIR = 'images/'
BATCH_SIZE = 32
NUM_CLASSES = 4


class ModelCreator:
    data = []
    labels = []

    def load_dataset(self, dataset_path: str = "dataset"):
        imagePaths = list(paths.list_images(dataset_path))
        data = []
        labels = []
        # loop over the image paths
        for imagePath in imagePaths:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]
            # load the input image (224x224) and preprocess it
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)
        # convert the data and labels to NumPy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = np.array(labels, dtype="float32")
        #labels = to_categorical(labels_c)

        return data, labels

    def build_model(self):
        baseModel = MobileNetV2(weights="imagenet", include_top=False,
                                input_shape=(224, 224, 3))
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(NUM_CLASSES, activation="softmax")(headModel)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)

        return model

    def build_custom_model(self):
        """model = Sequential([
            Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2,2),

            Conv2D(100, (3,3), activation='relu'),
            MaxPooling2D(2,2),

            Flatten(),
            Dropout(0.5),
            Dense(50, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])"""
        model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.summary()

        return model

    def create_augmentation_generator(self):

        def add_noise(img):
            '''Add random noise to an image'''
            VARIABILITY = 8
            deviation = VARIABILITY * random.random()
            noise = np.random.normal(0, deviation, img.shape)
            img += noise
            np.clip(img, 0., 255.)
            return img

        generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        generator2 = ImageDataGenerator(
            brightness_range=[0.2, 1.6],
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
            preprocessing_function=add_noise,
        )

        return generator

    def train_model(self, model, data: ndarray, labels: ndarray, image_generator: ImageDataGenerator):

        # callbacks
        # callbacks_list = [
        #     ModelCheckpoint(
        #         'weights/service_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
        #     EarlyStopping(monitor='val_accuracy', patience=5),
        #     ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        # ]

        callbacks_list = [
            ModelCheckpoint(
                'weights/service_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_accuracy', patience=8),
            ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        ]

        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                          test_size=0.20, stratify=labels, random_state=42)

        INIT_LR = 0.000125  # 1e-4
        EPOCHS = 100  # 20
        BATCH_SIZE = 32
        print("[INFO] compiling model...")
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the head of the network
        print("[INFO] training head...")
        history = model.fit(
            image_generator.flow(trainX, trainY, batch_size=BATCH_SIZE),
            steps_per_epoch=len(trainX) // BATCH_SIZE,
            validation_data=(testX, testY),
            validation_steps=len(testX) // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks_list)

        # To save the trained model
        model.save('mask_recognition_v4.h5')

        return history

    def show_history(self, history, epochs_number: int):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, epochs_number), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs_number), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs_number), history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs_number), history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")

        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        #
        # epochs_range = range(500)
        #
        # plt.figure(figsize=(15, 15))
        # plt.subplot(2, 2, 1)
        # plt.plot(epochs_range, acc, label='Training Accuracy')
        # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        # plt.legend(loc='lower right')
        # plt.title('Training and Validation Accuracy')
        #
        # plt.subplot(2, 2, 2)
        # plt.plot(epochs_range, loss, label='Training Loss')
        # plt.plot(epochs_range, val_loss, label='Validation Loss')
        # plt.legend(loc='upper right')
        # plt.title('Training and Validation Loss')
        # plt.show()



if __name__ == '__main__':
    model_creator = ModelCreator()
    data, labels = model_creator.load_dataset()
    model = model_creator.build_model()
    image_generator = model_creator.create_augmentation_generator()
    history = model_creator.train_model(model, data, labels, image_generator)
    model_creator.show_history(history, epochs_number=100)

    fasf = 32532
    asfqwa = 32532
