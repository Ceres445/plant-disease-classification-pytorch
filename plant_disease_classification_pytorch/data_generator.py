#!/usr/bin/env python

import os
import glob
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from .plant_dataset import PlantDataset

"""Responsible from generating data from datasets."""

train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)

valid_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)


def load_train_data(train_path, image_size, classes, max=5):
    images = []
    labels = []
    img_names = []
    class_array = []
    extension_list = ("*.jpg", "*.JPG")

    print("Going to read training images")
    for image_class in classes:
        index = classes.index(image_class)
        print("Now going to read {} files (Index: {})".format(image_class, index))
        for extension in extension_list:
            path = os.path.join(train_path, image_class, extension)
            files = glob.glob(path)
            for file_path in files[:max]:
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
                class_array.append(image_class)
    images = np.array(images)
    print(str.format("Completed reading {0} images of training dataset", len(images)))
    labels = np.array(labels)
    img_names = np.array(img_names)
    class_array = np.array(class_array)
    return images, labels, img_names, class_array


import numpy as np
from itertools import chain


def _indexing(x, indices):
    """
    :param x: array from which indices has to be fetched
    :param indices: indices to be fetched
    :return: sub-array from given array and indices
    """
    # np array indexing
    if hasattr(x, "shape"):
        return x[indices]

    # list indexing
    return [x[idx] for idx in indices]


def train_test_split(*arrays, test_size=0.25, shufffle=True, random_seed=1):
    """
    splits array into train and test data.
    :param arrays: arrays to split in train and test
    :param test_size: size of test set in range (0,1)
    :param shufffle: whether to shuffle arrays or not
    :param random_seed: random seed value
    :return: return 2*len(arrays) divided into train ans test
    """
    # checks
    assert 0 < test_size < 1
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length * test_size))
    n_train = length - n_test

    if shufffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(
        chain.from_iterable(
            (_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays
        )
    )


def read_datasets(train_path, image_size, classes, test_size):
    images, labels, img_names, class_array = load_train_data(
        train_path, image_size, classes
    )

    train_images, validation_images = train_test_split(images, test_size=test_size)
    train_labels, validation_labels = train_test_split(labels, test_size=test_size)
    train_img_names, validation_img_names = train_test_split(
        img_names, test_size=test_size
    )
    train_cls, validation_cls = train_test_split(class_array, test_size=test_size)

    train_dataset = PlantDataset(
        images=train_images,
        labels=train_labels,
        img_names=train_img_names,
        classes=train_cls,
        transform=train_transform,
    )
    validation_dataset = PlantDataset(
        images=validation_images,
        labels=validation_labels,
        img_names=validation_img_names,
        classes=validation_cls,
        transform=valid_transform,
    )

    return train_dataset, validation_dataset


def read_test_dataset(test_path, image_size):
    images = []
    img_names = []

    print("Going to read test images")
    files = os.listdir(test_path)
    count = 0
    for f in files:
        file_path = os.path.join(test_path, f)
        image = Image.open(file_path)
        image = image.resize((image_size, image_size))
        images.append(image)
        file_base = os.path.basename(file_path)
        img_names.append(file_base)
        count += 1
        if count % 5000 == 0:
            print(str.format("Read {0} test images", count))
    print(str.format("Completed reading {0} images of test dataset", count))

    img_names = np.array(img_names)

    test_dataset = PlantDataset(
        images=images,
        labels=None,
        img_names=img_names,
        classes=None,
        transform=test_transform,
    )
    return test_dataset
