from typing import Any

import cv2
from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from IPython.display import Image
from matplotlib import pyplot as plt
import numpy
from arm_vae import ARM_VAE


class vae_trainer:
    bin_train_loader: DataLoader[Any]

    def __init__(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        # setup device cuda vs. cpu
        # cuda = torch.cuda.is_available()
        self.cuda = False
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.epochs = 2
        self.LEARNING_RATE = 1e-3
        self.LATENT_SIZE = 20
        self.IMG_X = 640
        self.IMG_Y = 480
        self.model = None
        self.optimizer = None
        self.bin_train_loader = None
        self.current_epoch = 0

    def train_and_save_model(self, filename):
        print("train_and_save_model")
        self.__create_empty_model()
        self.__load_data()
        for self.current_epoch in range(1, self.epochs + 1):
            # train for one epoch
            train_loss = self.__train()
            # print the train loss for the epoch
            print(f'====> Epoch: {self.current_epoch} Average train loss: {train_loss / len(self.bin_train_loader.dataset):.4f}')
        # save the state dict of the model
        torch.save(self.model.state_dict(), filename)

    @staticmethod
    def __to_grey_scale(x):
        return numpy.asarray(x.convert("L")).astype(numpy.float64)/256.0

    def __load_data(self):
        self.bin_train_loader = DataLoader(ImageFolder('images',transform=self.__to_grey_scale))

    def __create_empty_model(self):
        print("create_empty_model")
        self.model = ARM_VAE(
            self.device,
            self.IMG_X, self.IMG_Y,
            self.LATENT_SIZE).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.LEARNING_RATE,
        )

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,x_hat, x, mu, logvar):

        # reconstruction loss (pushing the points apart)
        reconstruction_loss = nn.functional.mse_loss(
            x_hat, x.view(-1, self.IMG_Y * self.IMG_X), reduction='sum'
        )

        # KL divergence loss (the relative entropy between two distributions a multivariate gaussian and a normal)
        # (enforce a radius of 1 in each direction + pushing the means towards zero)
        kl_divergence_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        # print(f"logvar: {type(logvar)}, {logvar.shape}, ({logvar})")
        # print(f"logvar_exp: ({logvar.exp()})")
        # print(f"mu: {type(mu)}, ({mu})")
        # print(f"mu.pow: ({mu.pow(2)})")
        print(f"loss KLD={kl_divergence_loss}, MSE={reconstruction_loss}")

        return reconstruction_loss + kl_divergence_loss # we can use a beta parameter here (BCE + beta * KLD)

    # performs one epoch of training and returns the training loss for this epoch
    def __train(self):
        print("train")
        self.model.train()
        train_loss = 0
        image_num = 0
        for x, _ in self.bin_train_loader:
            x = x.to(self.device)
            # ===================forward=====================
            x_hat, mu, logvar = self.model(x)
            loss = self.loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()
            print(f" img={image_num} lbl={_[0]} epoch={self.current_epoch} loss={loss.item()} , total_loss={train_loss}")
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            image_num += 1
        return train_loss


# Defining main function
def main():
    print("train_vae main starting")
    trainer = vae_trainer()
    MODEL_FILE = 'train-all.pt'
    trainer.train_and_save_model(MODEL_FILE)


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
