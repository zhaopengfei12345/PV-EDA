import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
from torch import tensor
import torch.nn.init as init
import torch.nn.functional as F


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None
revgrad = RevGradFunc.apply


class RevGradLayer(Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


def cal_cheb_polynomial(laplacian, K):
    N = laplacian.size(0)
    I = torch.eye(N, device=laplacian.device, dtype=laplacian.dtype)

    cheb_polys = []
    cheb_polys.append(I)
    if K == 1:
        return torch.stack(cheb_polys, dim=0)
    cheb_polys.append(laplacian)
    if K == 2:
        return torch.stack(cheb_polys, dim=0)
    for k in range(2, K):
        T_k = 2 * torch.mm(laplacian, cheb_polys[-1]) - cheb_polys[-2]
        cheb_polys.append(T_k)

    return torch.stack(cheb_polys, dim=0)



def cal_laplacian(graph):
    I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
    graph = graph + I
    D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
    L = I - torch.mm(torch.mm(D, graph), D)
    return L


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        x_in = self.align(x)
        return torch.relu(x_gc + x_in)


class SpatioConvLayer_MLP(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super(SpatioConvLayer_MLP, self).__init__()
        self.ks = ks
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x = x.permute(0, 2, 1)
        x_c = torch.einsum("knm,bim->bikn", Lk, x)
        x_gc = torch.einsum("iok,bikn->bon", self.theta, x_c) + self.b
        return F.relu(x_gc)


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, device):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], c[1], device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x, graph):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1, graph)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)
