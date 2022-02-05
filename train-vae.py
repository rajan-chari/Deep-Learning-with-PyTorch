import cv2
from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from IPython.display import Image
from matplotlib import pyplot as plt
import numpy


class vae_trainer:
    def __init__(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        # setup device cuda vs. cpu
        # cuda = torch.cuda.is_available()
        self.cuda = False
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.LEARNING_RATE = 1e-3
        self.epochs = 50

    def train_and_save_model(self, filename):
        model = create_empty_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.LEARNING_RATE,
        )
        for epoch in range(1, epochs + 1):
            # train for one epoch
            train_loss = train(model, optimizer)
            # print the train loss for the epoch
            print(f'====> Epoch: {epoch} Average train loss: {train_loss / len(bin_train_loader.dataset):.4f}')
        # save the state dict of the model
        torch.save(model.state_dict(), filename)


# Defining main function
def main():
    print("hey there")


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
