#!/usr/bin/env python

import os
import glob
import numpy as np

from PIL import Image
from sklearn.utils import shuffle
from plant_disease_classification_pytorch.dataset import Dataset


def load_train_data(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    class_array = []
    extension_list = ('*.jpg', '*.JPG')

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        for extension in extension_list:
            path = os.path.join(train_path, fields, extension)
            files = glob.glob(path)
            for file_path in files:
                image = Image.open(file_path)
                image = image.resize((image_size, image_size))
                pixels = np.array(image)
                pixels = pixels.astype(np.float32)
                pixels = np.multiply(pixels, 1.0 / 255.0)
                images.append(pixels)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                file_base = os.path.basename(file_path)
                img_names.append(file_base)
                class_array.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    class_array = np.array(class_array)
    return images, labels, img_names, class_array


def read_train_sets(train_path, image_size, classes, validation_size):
    dataset = Dataset()

    images, labels, img_names, class_array = load_train_data(train_path, image_size, classes)
    images, labels, img_names, class_array = shuffle(images, labels, img_names, class_array)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = class_array[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = class_array[validation_size:]

    dataset.train = Dataset(train_images, train_labels, train_img_names, train_cls)
    dataset.validation = Dataset(validation_images, validation_labels, validation_img_names, validation_cls)

    return dataset
