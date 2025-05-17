import torch, os
from torch import nn
from torch.functional import F


## START: Adapted from https://github.com/iliaishacked/curse_recurse/blob/main/VAE_experiments.ipynb
class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(Encoder, self).__init__()
        self.c = c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super(Decoder, self).__init__()
        self.c = c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, capacity=64, latent_dims=20):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(capacity=capacity, latent_dims=latent_dims)
        self.decoder = Decoder(capacity=capacity, latent_dims=latent_dims)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def extract_features(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        return {"latent_mu": latent_mu, "latent_logvar": latent_logvar}  

    def latent_sample(self, mu, logvar, sampleme=False):
        if self.training or sampleme:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def vae_loss(recon_x, x, mu, logvar, variational_beta=1):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence


def train_vae_gen(vae, train_dataloader, save_dir, num_epochs=30, learning_rate=1e-3, device=torch.device("cuda:0")):

    vae = vae.to(device)

    # num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    # print('Number of parameters: %d' % num_params)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

    vae.train()

    train_loss_avg = []

    # print('Training begin...')
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in train_dataloader:

            image_batch = image_batch.to(device)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # reconstruction error
            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        if epoch == num_epochs - 1:
            with open(os.path.join(save_dir, "log.txt"), "a") as f:
                f.write(f"Average reconstruction error: {train_loss_avg[-1]:.6f}\n")
        # print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
    torch.save(vae.state_dict(), os.path.join(save_dir, "checkpoint.pth"))
    return

