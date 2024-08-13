import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, _ = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction

class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1, device=torch.device("cuda")):
        super(LSTMVAE, self).__init__()
        self.device = device
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(latent_size, hidden_size, input_size, num_layers)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.fc_z = nn.Linear(latent_size, hidden_size)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden, cell = self.encoder(x)
        hidden = hidden[-1]
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparametrize(mu, logvar)
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        hidden = (self.fc_z(z[:, 0, :]).unsqueeze(0), cell)
        x_hat = self.decoder(z, hidden)
        return x_hat, mu, logvar

    def loss_function(self, x_hat, x, mu, logvar):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss
