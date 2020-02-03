import tensorflow as tf
from abc import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time

class Singleton:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance

class PathClass(Singleton):

    def __init__(self):
        self._data_folder = "C:/Users/eotlr/data/blending/training/"
        self._result_folder ="C:/Users/eotlr/machine_learning_model/"
        self._validation_dir = "C:/Users/eotlr/data/blending/validation/"

    def get_data_folder(self):
        return self._data_folder

    def get_result_folder(self):
        return self._result_folder

    def get_validation_dir(self):
        return self._validation_dir


class abstract_machine_learning_class(metaclass=ABCMeta):
    def __init__(self):
        self._train_dir = PathClass.instance().get_data_folder()
        self._validation_dir = PathClass.instance().get_validation_dir()
        self._result_folder = PathClass.instance().get_result_folder()
        self._train_malware_dir = self._train_dir+"malware/"
        self._train_safe_dir = self._train_dir+"safe/"
        self._validation_malware_dir = self._validation_dir+"malware/"
        self._validation_safe_dir = self._validation_dir+"safe/"
        self._num_train_malware_dir = len(os.listdir(self._train_malware_dir))
        self._num_train_safe_dir = len(os.listdir(self._train_safe_dir))
        self._num_validation_malware_dir = len(os.listdir(self._validation_malware_dir))
        self._num_validation_safe_dir = len(os.listdir(self._validation_safe_dir))
        self._total_train = self._num_train_malware_dir+self._num_train_safe_dir
        self._total_validation = self._num_validation_malware_dir + self._num_validation_safe_dir
        self._img_size = (256,256)
        self._batch_size = 20
        self._epochs = 20
        self._train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,horizontal_flip=True)
        self._validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self._train_imgae_gen = self._train_image_generator.flow_from_directory(batch_size=self._batch_size, directory=self._train_dir, shuffle=True, target_size=self._img_size, class_mode="binary")
        self._validation_image_gen = self._validation_image_generator.flow_from_directory(batch_size=self._batch_size, directory=self._validation_dir,target_size=self._img_size, class_mode="binary")

    def learning(self):
        # self._train_image_generator.flow_from_directory(batch_size=self._batch_size, directory=self._train_dir, shuffle=True, target_size=self._img_size, class_mode="binary")
        # self._validation_image_generator.flow_from_directory(batch_size=self._batch_size, directory=self._validation_dir,target_size=self._img_size, class_mode="binary")
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Conv2D(16, 8, padding="valid", activation="relu",
        #                            input_shape=(self._img_size[0] ,self._img_size[1],3)),
        #     tf.keras.layers.MaxPool2D(),
        #     tf.keras.layers.Conv2D(32, 4, padding="valid", activation="relu"),
        #     tf.keras.layers.MaxPool2D(),
        #     # tf.keras.layers.Conv1D(filters=64,kernel_size=2,padding="valid", activation="relu", strides=2),
        #     # tf.keras.layers.MaxPool1D(),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(1024, activation="relu"),
        #     tf.keras.layers.Dense(1, activation="sigmoid")
        # ])
        # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        sample_training_images , _ = next(self._train_imgae_gen)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(20, (2,2) ,input_shape=(self._img_size[0],self._img_size[1],3), activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(2,2))
        model.add(tf.keras.layers.Conv2D(32, (2,2) , activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(2,2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100,activation="relu"))
        model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        history = model.fit_generator(
            self._train_imgae_gen,
            steps_per_epoch=self._batch_size,
            epochs=self._epochs,
            validation_data=self._validation_image_gen,
            validation_steps=self._batch_size
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(self._epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        saveName = time.strftime("%Y-%m-%d-%H-%M",time.localtime(time.time()))
        model.save(PathClass.instance().get_result_folder()+saveName+".h5")

# class Inception_v4(metaclass=abstract_machine_learning_class):
#     def __init__(self):
#         abstract_machine_learning_class.__init__(self=self)
#
#     def learning(self):
#         sample_traing_image, _ = next(self._train_imgae_gen)
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Conv2D(16, 8, padding="valid", activation="relu",
#                                    input_shape=(self._img_size[0] ,self._img_size[1],3)),
#             tf.keras.layers.MaxPool2D(),
#             tf.keras.layers.Conv2D(32, 4, padding="valid", activation="relu"),
#
#             tf.keras.layers.MaxPool2D(),
#             tf.keras.layers.Conv1D(filters=64,kernel_size=2,padding="valid", activation="relu", strides=2),
#             tf.keras.layers.MaxPool1D(),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(1024, activation="relu"),
#             tf.keras.layers.Dense(1, activation="sigmoid")
#         ])
#         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#
#


