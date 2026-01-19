import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import masked_mae_loss
from models.module import ST_encoder, CLUB, ST_encoder_with_MLP
from models.layers import RevGradLayer

class PVEDA(nn.Module):
    def __init__(
            self,
            args,
            adj,
            adj_m,
            embed_size=64,
            output_T_dim=52,
            output_dim=1,
            device="cuda"
    ):
        super(PVEDA, self).__init__()
        self.args = args
        self.adj = adj
        self.adj_m = adj_m
        self.time_labels = 52
        self.embed_size = embed_size
        T_dim = args.input_length - 4 * (3 - 1)
        temp_spatial_label = list(range(args.num_nodes))
        self.spatial_label = torch.tensor(temp_spatial_label, device=args.device)
        self.st_encoder4variant = ST_encoder(args.num_nodes, args.d_input, args.d_model, 3, 3,
                               [[args.d_model, args.d_model // 2, args.d_model],
                                [args.d_model, args.d_model // 2, args.d_model]], args.input_length, args.dropout,
                               args.device)
        self.st_encoder4invariant_m = ST_encoder_with_MLP(args.num_nodes, args.d_model, 3, args.dropout,
                                                          args.d_weather, args.input_length, args.device)
        self.st_encoder4invariant = ST_encoder(args.num_nodes, args.d_input, args.d_model, 3, 3,
                        [[args.d_model, args.d_model // 2, args.d_model],
                        [args.d_model, args.d_model // 2, args.d_model]], args.input_length, args.dropout,
                        args.device)
        self.st_encoder4variant_m = ST_encoder_with_MLP(args.num_nodes, args.d_model, 3,
                                                        args.dropout, args.d_weather,
                                                        args.input_length, args.device)
        self.node_embeddings_1 = nn.Parameter(torch.randn(3, args.num_nodes, embed_size), requires_grad=True)
        self.node_embeddings_2 = nn.Parameter(torch.randn(3, embed_size, args.num_nodes), requires_grad=True)
        self.node_embeddings_3 = nn.Parameter(torch.randn(3, args.num_nodes, embed_size), requires_grad=True)
        self.node_embeddings_4 = nn.Parameter(torch.randn(3, embed_size, args.num_nodes), requires_grad=True)
        self.variant_predict_conv_1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.variant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.invariant_predict_conv_1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.invariant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.relu = nn.ReLU()
        self.variant_tconv = nn.Conv2d(in_channels=T_dim, out_channels=1, kernel_size=(1, 1), bias=True)
        self.variant_end_temproal = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels)
        )
        self.variant_end_spacial = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes)
        )
        self.variant_end_congest = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2)
        )

        self.invariant_tconv = nn.Conv2d(in_channels=T_dim, out_channels=1, kernel_size=(1, 1), bias=True)
        self.invariant_end_temporal = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels)
        )
        self.invariant_end_spatial = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes)
        )
        self.invariant_end_congest = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2)
        )
        self.alpha_linear = nn.Linear(1, 1)
        self.beta_linear = nn.Linear(1, 1)
        self.revgrad = RevGradLayer()
        self.mask = torch.zeros([args.batch_size, args.d_input, args.input_length,
                                 args.num_nodes], dtype=torch.float).to(device)
        self.receptive_field = args.input_length + 8
        self.mse_loss = torch.nn.MSELoss()
        self.mi_net = CLUB(embed_size, embed_size, embed_size * 2)
        self.optimizer_mi_net = torch.optim.Adam(self.mi_net.parameters(), lr=args.lr_init)
        self.mae = masked_mae_loss(mask_value=-1.0)
        self.linear_layer = nn.Linear(11, 1)
        self.weather_fusion_w = nn.Parameter(torch.ones(args.d_weather), requires_grad=True)  # 对应 W_1...W_d
        self.weather_fusion_b = nn.Parameter(torch.zeros(1), requires_grad=True)             # 对应 b_M


    def forward(self, x, x_m):
        x = x.permute(0, 3, 1, 2)
        invariant_output = self.st_encoder4invariant(x, self.adj)
        B, D, T_enc, N = invariant_output.shape
        w = F.softmax(self.weather_fusion_w, dim=0)
        A_M = (self.adj_m * w.view(1, 1, -1)).sum(dim=2)
        adj_m = F.relu(A_M + self.weather_fusion_b)
        invariant_output_m_full = self.st_encoder4invariant_m(x_m, adj_m)
        invariant_output_m_full = invariant_output_m_full.permute(0, 2, 1, 3)
        H = invariant_output_m_full.shape[2]

        if H == T_enc:
            invariant_output_m = invariant_output_m_full
        elif H > T_enc:
            invariant_output_m = invariant_output_m_full[:, :, -T_enc:, :]
        else:
            repeat = T_enc - H
            last = invariant_output_m_full[:, :, -1:, :].repeat(1, 1, repeat, 1)
            invariant_output_m = torch.cat([invariant_output_m_full, last], dim=2)

        invariant_output = invariant_output + invariant_output_m
        invariant_output = invariant_output.permute(0, 2, 1, 3)

        adaptive_adj = F.softmax(
            F.relu(torch.bmm(self.node_embeddings_1, self.node_embeddings_2)), dim=1
        )
        adaptive_adj_m = F.softmax(
            F.relu(torch.bmm(self.node_embeddings_3, self.node_embeddings_4)), dim=1
        )
        variant_output = self.st_encoder4variant.variant_encode(x, adaptive_adj)
        variant_output_m_full = self.st_encoder4variant_m.variant_encode(x_m, adaptive_adj_m)
        variant_output_m_full = variant_output_m_full.permute(0, 2, 1, 3)
        Hm = variant_output_m_full.shape[2]
        if Hm == T_enc:
            variant_output_m = variant_output_m_full
        elif Hm > T_enc:
            variant_output_m = variant_output_m_full[:, :, -T_enc:, :]
        else:
            repeat = T_enc - Hm
            last = variant_output_m_full[:, :, -1:, :].repeat(1, 1, repeat, 1)
            variant_output_m = torch.cat([variant_output_m_full, last], dim=2)
        variant_output = variant_output + variant_output_m
        variant_output = variant_output.permute(0, 2, 1, 3)

        return invariant_output, variant_output


    def predict(self, z1, z2, x):
        out_1 = self.relu(self.invariant_predict_conv_1(z1))
        out_1 = out_1.permute(0, 2, 3, 1)
        out_1 = self.invariant_predict_conv_2(out_1)
        out_1 = out_1.permute(0, 3, 2, 1)
        out_2 = self.relu(self.variant_predict_conv_1(z2))
        out_2 = out_2.permute(0, 2, 3, 1)
        out_2 = self.variant_predict_conv_2(out_2)
        out_2 = out_2.permute(0, 3, 2, 1)
        alpha = self.alpha_linear(out_1)
        beta = self.beta_linear(out_2)
        temp = torch.stack([alpha, beta], dim=-1)
        temp = F.softmax(temp, dim=-1)
        alpha, beta = temp[..., 0], temp[..., 1]
        out_1 = out_1 * alpha
        out_2 = out_2 * beta
        out = out_2 + out_1

        return out, alpha, beta

    def variant_loss(self, z2, date, c):
        z2 = self.variant_tconv(z2).squeeze(1)
        z_temporal = z2.mean(2).squeeze()
        y_temporal = self.variant_end_temproal(z_temporal)
        loss_temproal = F.cross_entropy(y_temporal, date)
        z_spatial = z2.transpose(2, 1)
        y_spatial = self.variant_end_spacial(z_spatial)
        y_spatial = y_spatial.mean(0)
        loss_spacial = F.cross_entropy(y_spatial, self.spatial_label)
        z2_congest = z2.transpose(2, 1).unsqueeze(1)
        y_congest = self.variant_end_congest(z2_congest)
        loss_congest = self.mse_loss(y_congest, c)

        return (loss_spacial + loss_temproal + loss_congest) / 3.

    def invariant_loss(self, z1, date, c):
        z1_r = self.revgrad(z1)
        z1_r = self.invariant_tconv(z1_r).squeeze(1)
        z1_temporal = z1_r.mean(2).squeeze()
        y_temporal = self.invariant_end_temporal(z1_temporal)
        loss_temporal = F.cross_entropy(y_temporal, date)
        z1_spatial = z1_r.transpose(2, 1)
        y_spatial = self.invariant_end_spatial(z1_spatial)
        y_spatial = y_spatial.mean(0)
        loss_spatial = F.cross_entropy(y_spatial, self.spatial_label)
        z1_congest = z1_r.transpose(2, 1).unsqueeze(1)
        y_congest = self.invariant_end_congest(z1_congest)
        loss_congest = self.mse_loss(y_congest, c)

        return (loss_spatial + loss_temporal + loss_congest) / 3.

    def pred_loss(self, z1, z2, x, y_true, scaler):
        y_pred, alpha, beta = self.predict(z1, z2, x)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
        loss = self.mae(y_pred[..., 0], y_true[..., 0])
        return loss

    def calculate_loss(self, x, z1, z2, target, c, time_label, scaler, loss_weights, training=False):
        l1 = self.pred_loss(z1, z2, x, target, scaler)
        loss = 0
        sep_loss = [l1.item()]
        if training and self.args.MMI:
            z1_temp = z1.transpose(3, 2).mean(1).mean(1)
            z2_temp = z2.transpose(3, 2).mean(1).mean(1)
            self.mi_net.train()
            temp1 = z1_temp.detach()
            temp2 = z2_temp.detach()
            for i in range(5):
                self.optimizer_mi_net.zero_grad()
                mi_loss = self.mi_net.learning_loss(temp1, temp2)
                mi_loss.backward()
                self.optimizer_mi_net.step()
            self.mi_net.eval()
            l4 = 0.1 * self.mi_net(z1_temp, z2_temp)
            loss = loss + l4
        loss = loss + loss_weights[0] * l1
        l2 = self.variant_loss(z2, time_label, c)
        sep_loss.append(l2.item())
        loss = loss + loss_weights[1] * l2
        l3 = self.invariant_loss(z1, time_label, c)
        sep_loss.append(l3.item())
        loss = loss + loss_weights[2] * l3
        if not training:
            if self.args.lr_mode == 'only':
                loss = l1
            elif self.args.lr_mode == 'add':
                loss = l1 + l2

        return loss, sep_loss
