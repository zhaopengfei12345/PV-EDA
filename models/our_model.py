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
        self.embed_size = embed_size
        self.output_T_dim = output_T_dim
        self.output_dim = output_dim
        self.device = device
        self.num_nodes = args.num_nodes
        self.time_labels = output_T_dim
        self.pgr_levels = 6
        self.lambda_vclub = getattr(args, "lambda_vclub", 0.1)

        self.st_encoder4invariant = ST_encoder(
            args.num_nodes,
            args.d_input,
            args.d_model,
            3,
            3,
            [[args.d_model, args.d_model // 2, args.d_model],
             [args.d_model, args.d_model // 2, args.d_model]],
            args.input_length,
            args.dropout,
            args.device
        )
        self.st_encoder4variant = ST_encoder(
            args.num_nodes,
            args.d_input,
            args.d_model,
            3,
            3,
            [[args.d_model, args.d_model // 2, args.d_model],
             [args.d_model, args.d_model // 2, args.d_model]],
            args.input_length,
            args.dropout,
            args.device
        )
        self.st_encoder4invariant_m = ST_encoder_with_MLP(
            args.num_nodes,
            args.d_model,
            3,
            args.dropout,
            args.d_weather,
            args.input_length,
            args.device
        )
        self.st_encoder4variant_m = ST_encoder_with_MLP(
            args.num_nodes,
            args.d_model,
            3,
            args.dropout,
            args.d_weather,
            args.input_length,
            args.device
        )

        self.node_embeddings_q_1 = nn.Parameter(torch.randn(3, args.num_nodes, embed_size), requires_grad=True)
        self.node_embeddings_q_2 = nn.Parameter(torch.randn(3, embed_size, args.num_nodes), requires_grad=True)
        self.node_embeddings_m_1 = nn.Parameter(torch.randn(3, args.num_nodes, embed_size), requires_grad=True)
        self.node_embeddings_m_2 = nn.Parameter(torch.randn(3, embed_size, args.num_nodes), requires_grad=True)

        self.weather_fusion_w = nn.Parameter(torch.ones(args.d_weather), requires_grad=True)
        self.weather_fusion_b = nn.Parameter(torch.zeros(1), requires_grad=True)

        t_dim = args.input_length - 4 * (3 - 1)

        self.invariant_predict_conv_1 = nn.Conv2d(t_dim, output_T_dim, 1)
        self.invariant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.variant_predict_conv_1 = nn.Conv2d(t_dim, output_T_dim, 1)
        self.variant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)

        self.alpha_head = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1)
        )
        self.beta_head = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1)
        )

        self.variant_tproj = nn.Conv2d(t_dim, 1, kernel_size=(1, 1), bias=True)
        self.invariant_tproj = nn.Conv2d(t_dim, 1, kernel_size=(1, 1), bias=True)

        self.variant_time_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels)
        )
        self.variant_space_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes)
        )
        self.variant_pgr_head = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, self.pgr_levels)
        )

        self.invariant_time_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels)
        )
        self.invariant_space_head = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes)
        )
        self.invariant_pgr_head = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, self.pgr_levels)
        )

        self.revgrad = RevGradLayer()
        self.mae = masked_mae_loss(mask_value=-1.0)
        self.mi_net = CLUB(embed_size, embed_size, embed_size * 2)
        self.optimizer_mi_net = torch.optim.Adam(self.mi_net.parameters(), lr=args.lr_init)
        self.relu = nn.ReLU()

        self.register_buffer("spatial_label", torch.arange(args.num_nodes, dtype=torch.long))

    def _build_static_weather_adj(self):
        if self.adj_m.dim() == 3:
            w = F.softmax(self.weather_fusion_w, dim=0)
            a_m = (self.adj_m * w.view(1, 1, -1)).sum(dim=2)
            a_m = F.relu(a_m + self.weather_fusion_b)
            return a_m
        return self.adj_m

    def _align_weather_repr(self, z_m, t_enc):
        z_m = z_m.permute(0, 2, 1, 3)
        h = z_m.shape[2]
        if h == t_enc:
            return z_m
        if h > t_enc:
            return z_m[:, :, -t_enc:, :]
        repeat = t_enc - h
        last = z_m[:, :, -1:, :].repeat(1, 1, repeat, 1)
        return torch.cat([z_m, last], dim=2)

    def _make_adaptive_adj(self, emb1, emb2):
        adj = torch.bmm(emb1, emb2)
        adj = F.relu(adj)
        adj = F.softmax(adj, dim=-1)
        return adj

    def _prepare_time_target(self, time_label, batch_size):
        if time_label.dim() == 3:
            target = torch.argmax(time_label, dim=-1)
        elif time_label.dim() == 2:
            target = time_label.long()
        elif time_label.dim() == 1:
            base = torch.arange(self.output_T_dim, device=time_label.device).unsqueeze(0).repeat(batch_size, 1)
            target = base.long()
        else:
            target = torch.arange(self.output_T_dim, device=self.device).unsqueeze(0).repeat(batch_size, 1).long()
        if target.size(1) != self.output_T_dim:
            if target.size(1) > self.output_T_dim:
                target = target[:, :self.output_T_dim]
            else:
                repeat = self.output_T_dim - target.size(1)
                last = target[:, -1:].repeat(1, repeat)
                target = torch.cat([target, last], dim=1)
        return target

    def _prepare_pgr_target(self, c, batch_size):
        if c.dim() == 4:
            target = torch.argmax(c, dim=-1)
        elif c.dim() == 3:
            target = c.long()
        elif c.dim() == 2:
            target = c.long().unsqueeze(-1).repeat(1, self.output_T_dim, self.num_nodes)
        else:
            target = torch.zeros(batch_size, self.output_T_dim, self.num_nodes, device=self.device, dtype=torch.long)
        target = target.clamp(min=0, max=self.pgr_levels - 1)
        if target.size(1) != self.output_T_dim:
            if target.size(1) > self.output_T_dim:
                target = target[:, :self.output_T_dim, :]
            else:
                repeat = self.output_T_dim - target.size(1)
                last = target[:, -1:, :].repeat(1, repeat, 1)
                target = torch.cat([target, last], dim=1)
        if target.size(2) != self.num_nodes:
            if target.size(2) > self.num_nodes:
                target = target[:, :, :self.num_nodes]
            else:
                repeat = self.num_nodes - target.size(2)
                last = target[:, :, -1:].repeat(1, 1, repeat)
                target = torch.cat([target, last], dim=2)
        return target

    def _prediction_head(self, z, conv1, conv2):
        out = self.relu(conv1(z))
        out = out.permute(0, 2, 3, 1)
        out = conv2(out)
        out = out.permute(0, 3, 2, 1)
        return out

    def _global_weights(self, z1, z2):
        h1 = z1.mean(dim=(1, 2, 3))
        h2 = z2.mean(dim=(1, 2, 3))
        h1 = h1.unsqueeze(-1).repeat(1, self.embed_size)
        h2 = h2.unsqueeze(-1).repeat(1, self.embed_size)
        a1 = self.alpha_head(h1)
        a2 = self.beta_head(h2)
        logits = torch.cat([a1, a2], dim=-1)
        weights = F.softmax(logits, dim=-1)
        alpha = weights[:, 0].view(-1, 1, 1, 1)
        beta = weights[:, 1].view(-1, 1, 1, 1)
        return alpha, beta

    def forward(self, x, x_m):
        x = x.permute(0, 3, 1, 2)

        invariant_q = self.st_encoder4invariant(x, self.adj)
        b, d, t_enc, n = invariant_q.shape
        static_adj_m = self._build_static_weather_adj()
        invariant_m = self.st_encoder4invariant_m(x_m, static_adj_m)
        invariant_m = self._align_weather_repr(invariant_m, t_enc)
        invariant_output = invariant_q + invariant_m
        invariant_output = invariant_output.permute(0, 2, 1, 3)

        adaptive_adj_q = self._make_adaptive_adj(self.node_embeddings_q_1, self.node_embeddings_q_2)
        adaptive_adj_m = self._make_adaptive_adj(self.node_embeddings_m_1, self.node_embeddings_m_2)
        variant_q = self.st_encoder4variant.variant_encode(x, adaptive_adj_q)
        variant_m = self.st_encoder4variant_m.variant_encode(x_m, adaptive_adj_m)
        variant_m = self._align_weather_repr(variant_m, t_enc)
        variant_output = variant_q + variant_m
        variant_output = variant_output.permute(0, 2, 1, 3)

        return invariant_output, variant_output

    def predict(self, z1, z2, x):
        out_1 = self._prediction_head(z1, self.invariant_predict_conv_1, self.invariant_predict_conv_2)
        out_2 = self._prediction_head(z2, self.variant_predict_conv_1, self.variant_predict_conv_2)
        alpha, beta = self._global_weights(z1, z2)
        out = alpha * out_1 + beta * out_2
        return out, alpha, beta

    def _ssl_variant(self, z2, time_label, c):
        b = z2.shape[0]
        z2p = self.variant_tproj(z2).squeeze(1)

        time_feat = z2p.mean(dim=2)
        time_logits = self.variant_time_head(time_feat)
        time_target = self._prepare_time_target(time_label, b)
        loss_time = F.cross_entropy(
            time_logits.reshape(-1, self.time_labels),
            time_target.reshape(-1)
        )

        space_feat = z2p.transpose(1, 2)
        space_logits = self.variant_space_head(space_feat)
        spatial_target = self.spatial_label.unsqueeze(0).repeat(b, 1)
        loss_space = F.cross_entropy(
            space_logits.reshape(-1, self.num_nodes),
            spatial_target.reshape(-1)
        )

        pgr_feat = z2p.permute(0, 2, 1)
        pgr_logits = self.variant_pgr_head(pgr_feat)
        pgr_target = self._prepare_pgr_target(c, b)
        loss_pgr = F.cross_entropy(
            pgr_logits.reshape(-1, self.pgr_levels),
            pgr_target.reshape(-1)
        )

        return (loss_space + loss_time + loss_pgr) / 3.0

    def _ssl_invariant(self, z1, time_label, c):
        b = z1.shape[0]
        z1r = self.revgrad(z1)
        z1r = self.invariant_tproj(z1r).squeeze(1)

        time_feat = z1r.mean(dim=2)
        time_logits = self.invariant_time_head(time_feat)
        time_target = self._prepare_time_target(time_label, b)
        loss_time = F.cross_entropy(
            time_logits.reshape(-1, self.time_labels),
            time_target.reshape(-1)
        )

        space_feat = z1r.transpose(1, 2)
        space_logits = self.invariant_space_head(space_feat)
        spatial_target = self.spatial_label.unsqueeze(0).repeat(b, 1)
        loss_space = F.cross_entropy(
            space_logits.reshape(-1, self.num_nodes),
            spatial_target.reshape(-1)
        )

        pgr_feat = z1r.permute(0, 2, 1)
        pgr_logits = self.invariant_pgr_head(pgr_feat)
        pgr_target = self._prepare_pgr_target(c, b)
        loss_pgr = F.cross_entropy(
            pgr_logits.reshape(-1, self.pgr_levels),
            pgr_target.reshape(-1)
        )

        return (loss_space + loss_time + loss_pgr) / 3.0

    def pred_loss(self, z1, z2, x, y_true, scaler):
        y_pred, _, _ = self.predict(z1, z2, x)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
        loss = self.mae(y_pred[..., 0], y_true[..., 0])
        return loss

    def calculate_loss(self, x, z1, z2, target, c, time_label, scaler, loss_weights, training=False):
        l1 = self.pred_loss(z1, z2, x, target, scaler)
        l2 = self._ssl_variant(z2, time_label, c)
        l3 = self._ssl_invariant(z1, time_label, c)

        loss = loss_weights[0] * l1 + loss_weights[1] * l2 + loss_weights[2] * l3

        if self.args.MMI:
            z1_temp = z1.mean(dim=(1, 2, 3)).unsqueeze(-1).repeat(1, self.embed_size)
            z2_temp = z2.mean(dim=(1, 2, 3)).unsqueeze(-1).repeat(1, self.embed_size)
            if training:
                self.mi_net.train()
                temp1 = z1_temp.detach()
                temp2 = z2_temp.detach()
                for _ in range(5):
                    self.optimizer_mi_net.zero_grad()
                    mi_learning_loss = self.mi_net.learning_loss(temp1, temp2)
                    mi_learning_loss.backward()
                    self.optimizer_mi_net.step()
            self.mi_net.eval()
            l4 = self.lambda_vclub * self.mi_net(z1_temp, z2_temp)
            loss = loss + l4

        if not training:
            if self.args.lr_mode == 'only':
                loss = l1
            elif self.args.lr_mode == 'add':
                loss = l1 + l2 + l3

        sep_loss = [l1.item(), l2.item(), l3.item()]
        return loss, sep_loss
