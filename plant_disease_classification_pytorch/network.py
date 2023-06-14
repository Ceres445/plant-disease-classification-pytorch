#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from plant_disease_classification_pytorch import data_generator, constant

IMAGE_SIZE = 128
BATCH_SIZE = 25
LEARNING_RATE = 0.001
EPOCHS = 35

# CPU or GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PlantDiseaseModel(nn.Module):
    """Convolutional Neural Network which does the raining."""

    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 30 * 30, 1024)
        self.fc2 = nn.Linear(1024, constant.NUMBER_OF_CLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    @staticmethod
    def create_dataloaders():
        train_dataset, validation_dataset = data_generator.read_datasets(
            constant.TRAINING_SET_PATH, IMAGE_SIZE, constant.classes(), 0.2
        )
        # test_dataset = data_generator.read_test_dataset(constant.TEST_SET_PATH, IMAGE_SIZE)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        valid_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        # test_loader = DataLoader(
        #     dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        # )

        return train_loader, valid_loader

    def train_model(self, epochs, train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        for epoch in range(1, epochs + 1):
            # keep track of training and validation loss

            # training the model
            self.train()
            for data, target in train_loader:
                # move tensors to GPU
                data = data.to(DEVICE)
                target = target.to(DEVICE)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self(data)
                # calculate the batch loss
                target = torch.max(target, 1)[1]
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss wrt model parameters
                loss.backward()
                # perform a ingle optimization step (parameter update)
                optimizer.step()

    def test_model(self, testloader: DataLoader):
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        valid_loss = 0.0
        self.eval()
        with torch.no_grad():
            for data, labels in testloader:
                data = data.to(device=DEVICE)
                labels = labels.to(device=DEVICE)

                outputs = self(data)
                _, predicted = torch.max(outputs.data, 0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        for data, target in testloader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = self(data)

            target = torch.max(target, 1)[1]
            loss = criterion(output, target)

            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        valid_loss = valid_loss / len(testloader.sampler)

        return (100 * correct / total, valid_loss)
