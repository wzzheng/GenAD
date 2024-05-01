
import torch
import torch.nn as nn

from mmdet.models import LOSSES

from .layers import Bottleneck


class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
        self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.encoder = DistributionEncoder1DV2(
            in_channels,
            self.compress_dim,
        )

        self.last_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
        )

    def forward(self, s_t):
        encoding = self.encoder(s_t.permute(0, 2, 1))
        mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

class DistributionEncoder2D(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            Bottleneck(in_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
        )

    def forward(self, s_t):
        return self.model(s_t)

class DistributionEncoder1D(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=in_channels*2, kernel_size=1, stride=1),
            nn.Conv1d(in_channels*2, out_channels=in_channels*2, kernel_size=1, stride=1),
            nn.Conv1d(in_channels*2, out_channels=in_channels, kernel_size=1, stride=1),
            nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        )

    def forward(self, s_t):
        return self.model(s_t)

class DistributionEncoder1DV2(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels * 2, out_channels=in_channels * 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s_t):
        s_t = self.relu(self.conv1(s_t))
        s_t = self.relu(self.conv2(s_t))
        s_t = self.conv3(s_t)

        return s_t

class DistributionDecoder1DV2(nn.Module):
    """Decodes sample to future states.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels=in_channels * 8, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels * 8, out_channels=in_channels * 8, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels * 8, out_channels=out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_t):
        f_t = self.relu(self.conv1(f_t))
        f_t = self.relu(self.conv2(f_t))
        f_t = self.conv3(f_t)

        return f_t

class PredictModel(nn.Module):
    """predict future states with rnn.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels*2)
        self.linear2 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.linear3 = nn.Linear(hidden_channels*4, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x , h):
        x, h = self.gru(x, h)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PredictModelHidden(nn.Module):
    """predict future states with rnn.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels*2)
        self.linear2 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.linear3 = nn.Linear(hidden_channels*4, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x, h = self.gru(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x




@LOSSES.register_module()
class ProbabilisticLoss(nn.Module):
    """kl-loss for present distribution and future distribution.
    """
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1)) * self.loss_weight

        return kl_loss






