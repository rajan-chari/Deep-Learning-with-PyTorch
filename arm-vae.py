class ARM_VAE(nn.Module):
    def __init__(self, device, img_x, img_y, latent_size):
        super().__init__()
        self.__device = device
        self.__latent_size = latent_size
        self.IMG_X = img_x
        self.IMG_Y = img_y

        self.encoder = nn.Sequential(
            nn.Linear(self.IMG_Y * self.IMG_X, self.__latent_size ** 2).double(),
            nn.ReLU().double(),
            nn.Linear(self.__latent_size ** 2, self.__latent_size * 2).double()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.__latent_size, self.__latent_size ** 2).double(),
            nn.ReLU().double(),
            nn.Linear(self.__latent_size ** 2, self.IMG_Y * self.IMG_X).double(),
            nn.Sigmoid().double(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        mu_logvar = self.encoder(x.view(-1, self.IMG_Y * self.IMG_X)).view(-1, 2, self.__latent_size)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, n_samples):
        z = torch.randn((n_samples, self.__latent_size)).to(self.__device)
        return self.decode(z)

    def to_device(self):
        self.to(self.__device)
