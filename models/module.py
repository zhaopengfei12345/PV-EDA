import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.layers import STConvBlock, cal_cheb_polynomial, cal_laplacian, SpatioConvLayer_MLP

class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class ST_encoder(nn.Module):
    def __init__(self, num_nodes, d_input, d_output, Ks, Kt, blocks, input_window, drop_prob, device):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = d_output
        self.output_dim = d_output
        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.input_window = input_window
        self.output_window = 1
        self.drop_prob = drop_prob
        self.blocks[0][0] = self.feature_dim
        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock')
        self.device = device
        self.input_conv = nn.Conv2d(d_input, d_output, 1)
        self.st_conv1 = STConvBlock(self.Ks, self.Kt, self.num_nodes, self.blocks[0], self.drop_prob, self.device)
        self.st_conv2 = STConvBlock(self.Ks, self.Kt, self.num_nodes, self.blocks[1], self.drop_prob, self.device)

    def forward(self, x, graph):
        lap_mx = cal_laplacian(graph)
        Lk = cal_cheb_polynomial(lap_mx, self.Ks)
        x = self.input_conv(x)
        x_st1 = self.st_conv1(x, Lk)
        x_st2 = self.st_conv2(x_st1, Lk)
        return x_st2

    def variant_encode(self, x, graph):
        x = self.input_conv(x)
        x_st1 = self.st_conv1(x, graph)
        x_st2 = self.st_conv2(x_st1, graph)
        return x_st2


class ST_encoder_with_MLP(nn.Module):
    def __init__(self, num_nodes, d_output, Ks, drop_prob, m_dim, input_length, device):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = d_output
        self.output_dim = d_output
        self.Ks = Ks
        self.drop_prob = drop_prob
        self.m_dim = m_dim
        self.device = device
        self.mlp_in = nn.Sequential(
            nn.Linear(self.m_dim, d_output),
            nn.ReLU(),
            nn.Linear(d_output, d_output)
        )
        self.spatial_conv = SpatioConvLayer_MLP(self.Ks, d_output, d_output, device)
        self.mlp_out = nn.Sequential(
            nn.Linear(d_output, d_output),
            nn.ReLU(),
            nn.Linear(d_output, d_output)
        )

        self.dropout = nn.Dropout(drop_prob)

    def _encode_with_Lk(self, x_m, Lk):
        if x_m.dim() == 3:
            x_m = x_m.unsqueeze(1)
        B, H, N, F = x_m.shape
        x = self.mlp_in(x_m)
        x = x.view(B * H, N, self.feature_dim)
        x = self.spatial_conv(x, Lk)
        x = x.view(B, H, self.feature_dim, N)
        x = x.permute(0, 1, 3, 2)
        x = self.mlp_out(x)
        x = x.permute(0, 1, 3, 2)

        return self.dropout(x)

    def forward(self, x_m, graph):
        lap_mx = cal_laplacian(graph)
        Lk = cal_cheb_polynomial(lap_mx, self.Ks)
        return self._encode_with_Lk(x_m, Lk)

    def variant_encode(self, x_m, Lk):
        return self._encode_with_Lk(x_m, Lk)
