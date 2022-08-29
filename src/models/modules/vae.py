import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(0.2)
        self.norm_layer = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = self.LeakyReLU(self.layer1(x))
        h = self.norm_layer(h)
        h = self.dropout_layer(h)

        h = self.LeakyReLU(self.layer2(h))
        h = self.norm_layer(h)
        h = self.dropout_layer(h)

        h = self.LeakyReLU(self.layer3(h))
        h = self.dropout_layer(h)

        mean = self.mean_layer(h)
        log_var = self.var_layer(h)

        return h, mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.layer1 = nn.Linear(latent_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(0.2)
        self.norm_layer = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = self.LeakyReLU(self.layer1(x))
        h = self.norm_layer(h)
        h = self.dropout_layer(h)

        h = self.LeakyReLU(self.layer2(h))
        h = self.norm_layer(h)
        h = self.dropout_layer(h)

        output = torch.sigmoid(self.layer3(h))
        return output


class VAENet(nn.Module):
    def __init__(
        self,
        encoder_input_dim=104,
        encoder_hidden_dim=512,
        encoder_latent_dim=32,
        decoder_latent_dim=32,
        decoder_hidden_dim=512,
        decoder_output_dim=104,
        *args,
        **kwargs,
    ):
        super(VAENet, self).__init__()
        self.Encoder = Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=encoder_latent_dim,
        )
        self.Decoder = Decoder(
            latent_dim=decoder_latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=decoder_output_dim,
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        latent, mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)

        return latent, x_hat, mean, log_var


class LossFunctions:
    def __init__(self, w_rec, w_kl):
        self.w_rec = w_rec
        self.w_kl = w_kl

    def cal_loss(self, x, x_hat, mean, log_var):
        reconstruction_loss = self.reconstruction_loss(x, x_hat)
        kl_loss = self.kl_loss(mean, log_var)
        loss = self.w_rec * reconstruction_loss + self.w_kl * kl_loss
        return reconstruction_loss * self.w_rec, kl_loss * self.w_kl, loss

    def reconstruction_loss(self, x, x_hat):
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        return reconstruction_loss.mean()

    def kl_loss(self, mean, log_var):
        kl_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_loss.mean()

