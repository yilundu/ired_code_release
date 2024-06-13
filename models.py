from diffusion_lib.nlm import LogicMachine
from diffusion_lib.transformer import GPT
import einops
import torch.nn as nn
from torch import einsum
import torch
import math
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend()

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class ResBlock(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, time_dim=64):
        super(ResBlock, self).__init__()

        self.filters = filters
        self.downsample = downsample

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=2)

        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        self.time_fc1 = nn.Linear(time_dim, 2*filters)
        self.time_fc2 = nn.Linear(time_dim, 2*filters)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, time_latent):
        x_orig = x

        latent_1 = self.time_fc1(time_latent)
        latent_2 = self.time_fc2(time_latent)

        gain = latent_1[:, :self.filters, None, None]
        bias = latent_1[:, self.filters:, None, None]

        gain2 = latent_2[:, :self.filters, None, None]
        bias2 = latent_2[:, self.filters:, None, None]

        x = self.conv1(x)
        # x = (gain + 1) * x + bias
        x = swish(x)

        x = self.conv2(x)
        x = (gain2 + 1) * x + bias2
        x = swish(x)

        x_out = x_orig + x

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def swish(x):
    return x * torch.sigmoid(x)


class EBM(nn.Module):
    def __init__(self, inp_dim, out_dim, is_ebm: bool = True):
        super(EBM, self).__init__()
        h = 512

        fourier_dim, time_dim = 128, 128

        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.fc1 = nn.Linear(inp_dim + out_dim, h)
        self.is_ebm = is_ebm

        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim if is_ebm else out_dim)

        self.t_map_fc2 = nn.Linear(time_dim, 2 * h)
        self.t_map_fc3 = nn.Linear(time_dim, 2 * h)

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_ebm = is_ebm

    def forward(self, *args):
        if self.is_ebm:
            x, t = args
        else:
            x, y, t = args
            x = torch.cat([x, y], dim=-1)

        t_emb = self.time_mlp(t)

        fc2_gain, fc2_bias = torch.chunk(self.t_map_fc2(t_emb), 2, dim=-1)
        fc3_gain, fc3_bias = torch.chunk(self.t_map_fc3(t_emb), 2, dim=-1)

        h = swish(self.fc1(x))
        h = swish(self.fc2(h) * (fc2_gain + 1) + fc2_bias)
        h = swish(self.fc3(h) * (fc3_gain + 1) + fc3_bias)

        if self.is_ebm:
            output = self.fc4(h).pow(2).sum(dim=-1)[..., None]
        else:
            output = self.fc4(h)

        return output


class AutoencodeModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        assert inp_dim == out_dim == 729
        h = 128

        self.conv1 = nn.Conv2d(9, h, 3, padding=1)
        self.conv2 = nn.Conv2d(h, h, 3, padding=1)
        self.conv3 = nn.Conv2d(h, 3, 3, padding=1)

        self.conv4 = nn.Conv2d(3, h, 3, padding=1)
        self.conv5 = nn.Conv2d(h, h, 3, padding=1)

        self.conv6 = nn.Conv2d(h, 9, 1, padding=0)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def decode(self, latent):
        h = latent

        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        output = self.conv6(h)

        return output

    def forward(self, inp, return_latent=False):
        x = einops.rearrange(inp, 'b (h w c) -> b c h w', h=9, w=9).contiguous()

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)

        latent = h
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        output = self.conv6(h)

        if return_latent:
            return output, latent
        else:
            return output


class SudokuLatentEBM(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        fourier_dim, time_dim = 128, 64
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        h = 128
        self.conv1 = nn.Conv2d(9 + 3, h, 3, padding=1)

        self.res1a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res1b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn1 = Attention(h, dim_head=128)

        self.res2a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res2b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn2 = Attention(h, dim_head=128)

        self.res3a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res3b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn3 = Attention(h, dim_head=128)

        self.conv5 = nn.Conv2d(h, 3, 1, padding=0)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        inp = x[..., :729]
        out = x[..., 729:]

        x = einops.rearrange(inp, 'b (h w c) -> b c h w', h=9, w=9)
        y = einops.rearrange(out, 'b (h w c) -> b c h w', h=9, w=9)
        x = torch.cat((x, y), dim=1)

        h = swish(self.conv1(x))

        h = self.res1a(h, t_emb)
        h = self.res1b(h, t_emb)
        # h = self.attn1(h)
        h = self.res2a(h, t_emb)
        h = self.res2b(h, t_emb)
        # h = self.attn2(h)
        h = self.res3a(h, t_emb)
        h = self.res3b(h, t_emb)
        # h = self.attn3(h)

        output = self.conv5(h)
        energy = output.pow(2).sum(dim=1).sum(dim=1).sum(dim=1)[:, None]

        return energy


class SudokuEBM(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        assert inp_dim == out_dim == 729

        fourier_dim, time_dim = 128, 64
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        h = 384
        self.conv1 = nn.Conv2d(9 + 9, h, 3, padding=1)

        self.res1a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res1b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn1 = Attention(h, dim_head=128)

        self.res2a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res2b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn2 = Attention(h, dim_head=128)

        self.res3a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res3b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn3 = Attention(h, dim_head=128)

        self.conv5 = nn.Conv2d(h, 9, 1, padding=0)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        inp, out = torch.chunk(x, 2, dim=-1)

        x = einops.rearrange(inp, 'b (h w c) -> b c h w', h=9, w=9)
        y = einops.rearrange(out, 'b (h w c) -> b c h w', h=9, w=9)
        x = torch.cat((x, y), dim=1)

        h = swish(self.conv1(x))

        h = self.res1a(h, t_emb)
        h = self.res1b(h, t_emb)
        # h = self.attn1(h)
        h = self.res2a(h, t_emb)
        h = self.res2b(h, t_emb)
        # h = self.attn2(h)
        h = self.res3a(h, t_emb)
        h = self.res3b(h, t_emb)
        # h = self.attn3(h)

        output = self.conv5(h)
        # energy = (output - y).pow(2).sum(dim=1).sum(dim=1).sum(dim=1)[:, None]
        # return energy

        output = output.pow(2).sum(dim=[1, 2, 3])[:, None]
        return output


class SudokuDenoise(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        assert inp_dim == out_dim == 729

        fourier_dim, time_dim = 128, 64
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        h = 128

        height, width = 9, 9
        y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))

        pos_offset = torch.stack([x_grid, y_grid], dim=0)
        self.pos_offset = pos_offset[None, :, :, :]

        self.conv1 = nn.Conv2d(9 + 9 + 2, h, 3, padding=1)

        self.res1a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res1b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn1 = Attention(h, dim_head=128)

        self.res2a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res2b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn2 = Attention(h, dim_head=128)

        self.res3a = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)
        self.res3b = ResBlock(downsample=False, rescale=False, filters=h, time_dim=time_dim)

        self.attn3 = Attention(h, dim_head=128)

        self.conv5 = nn.Conv2d(h, 9, 1, padding=0)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, inp, out, t):
        t_emb = self.time_mlp(t)

        pos_offset = self.pos_offset.expand(inp.size(0), -1, -1, -1).to(inp.device)
        x = einops.rearrange(inp, 'b (h w c) -> b c h w', h=9, w=9)
        y = einops.rearrange(out, 'b (h w c) -> b c h w', h=9, w=9)
        x = torch.cat((x, y, pos_offset), dim=1)

        h = swish(self.conv1(x))

        h = self.res1a(h, t_emb)
        h = self.res1b(h, t_emb)
        h = self.attn1(h)

        h = self.res2a(h, t_emb)
        h = self.res2b(h, t_emb)
        h = self.attn2(h)

        h = self.res3a(h, t_emb)
        h = self.res3b(h, t_emb)
        h = self.attn3(h)

        output = self.conv5(h)
        output = einops.rearrange(output, 'b c h w -> b (h w c)')
        return output


class SudokuTransformerEBM(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        assert inp_dim == out_dim == 729

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        nr_layers = 1
        nr_heads = 4
        hidden_dim = 128

        fourier_dim = 128
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_embedding = nn.Linear(18, hidden_dim)
        self.transformer = GPT(nr_layers=nr_layers, nr_heads=nr_heads, hidden_dim=hidden_dim, output_dim=9, max_length=81)

    def forward(self, x, t):
        inp, out = torch.chunk(x, 2, dim=-1)
        t_emb = self.time_mlp(t)

        x = einops.rearrange(inp, 'b (h w c) -> b (h w) c', h=9, w=9)
        y = einops.rearrange(out, 'b (h w c) -> b (h w) c', h=9, w=9)
        x = torch.cat((x, y), dim=-1)

        h = self.input_embedding(x) + t_emb[:, None, :]
        h = self.transformer(h)

        energy = (y - h).pow(2).sum(dim=-1).sum(dim=-1)[:, None]
        # energy = h.squeeze(-1).sum(dim=-1)
        return energy


class GraphEBM(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        fourier_dim, time_dim = 512, 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        h = 32
        self.nlm = LogicMachine(2, 3, [0, 0, inp_dim + 1 + time_dim, 0], [h, h, h, h], None)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, inp, out, t):
        # cond.shape == [B, N, N, 4]
        # x.shape == [B, N, N]
        t_emb = self.time_mlp(t)
        t_emb = einops.repeat(t_emb, 'b c -> b n1 n2 c', n1=out.shape[1], n2=out.shape[2])

        x = torch.concat([inp, out.unsqueeze(-1), t_emb], dim=-1)
        x = self.nlm([None, None, x, None])
        x = self.fc_out(x[2]).squeeze(-1)
        x = x.pow(2).sum(dim=[1, 2])[:, None]
        return x


class GraphReverse(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim

        fourier_dim, time_dim = 512, 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        h = 32
        self.nlm = LogicMachine(2, 3, [0, 0, inp_dim + 1 + time_dim, 0], [h, h, h, h], None)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, inp, out, t):
        # cond.shape == [B, N, N, 4]
        # x.shape == [B, N, N]
        t_emb = self.time_mlp(t)
        t_emb = einops.repeat(t_emb, 'b c -> b n1 n2 c', n1=out.shape[1], n2=out.shape[2])

        x = torch.concat([inp, out.unsqueeze(-1), t_emb], dim=-1)
        x = self.nlm([None, None, x, None])
        x = self.fc_out(x[2]).squeeze(-1)
        return x

    def randn(self, batch_size, shape, inp, device):
        assert shape[0] == 1
        return torch.randn((batch_size, *inp.shape[1:-1]), device=device)


class NLMConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.nlm = LogicMachine(2, 3, [0, 0, inp_dim, 0], [out_dim, out_dim, out_dim, out_dim], None)
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # X shape is [B, T, N, N, C]
        b, t, n, _, c = x.shape
        x = einops.rearrange(x, 'b t n1 n2 c -> (b t) n1 n2 c')
        x = self.nlm([None, None, x, None])[2]
        x = swish(x)
        x = einops.rearrange(x, '(b t) n1 n2 c -> (b n1 n2) c t', b=b, t=t)
        x = self.conv(x)
        x = einops.rearrange(x, '(b n1 n2) c t -> b t n1 n2 c', b=b, n1=n, n2=n)
        x = swish(x).contiguous()
        return x


class NLMConv1DBlock(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.nlm = LogicMachine(2, 2, [0, inp_dim, 3], [out_dim, out_dim, out_dim], None)
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # X shape is [B, T, N, C]
        b, t, n, c = x.shape
        x = einops.rearrange(x, 'b t n c -> (b t) n c')

        # generate the comparison matrix
        arange = torch.arange(n)
        # generate a binary comparison i < j
        cmp1 = arange[None, :, None] < arange[None, None, :]
        cmp2 = arange[None, :, None] == arange[None, None, :]
        cmp = torch.stack([cmp1, cmp2], dim=-1).float()
        cmp = cmp.repeat(b * t, 1, 1, 1).to(x.device)

        numbers = x[:, :, 1]
        cmp3 = numbers[:, :, None] < numbers[:, None, :]
        cmp = torch.concat([cmp, cmp3.unsqueeze(-1)], dim=-1)

        x = self.nlm([None, x, cmp])[1]
        x = swish(x)
        x = einops.rearrange(x, '(b t) n c -> (b n) c t', b=b, t=t)
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c t -> b t n c', b=b, n=n)
        x = swish(x).contiguous()
        return x


class GNNConvEBM(nn.Module):
    def __init__(self, inp_dim, out_dim, use_1d: bool = False):
        super().__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.use_1d = use_1d

        fourier_dim, time_dim = 512, 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        h = inp_dim + out_dim + time_dim

        if self.use_1d:
            self.block1 = NLMConv1DBlock(h, 32)
            self.block2 = NLMConv1DBlock(32, 32)
        else:
            self.block1 = NLMConvBlock(h, 32)
            self.block2 = NLMConvBlock(32, 32)
            self.block3 = NLMConvBlock(32, 32)

        self.fc_out = nn.Linear(32, 1)

    def forward(self, inp, out, t):
        # inp.shape == [B, N, N, inp_dim]
        # out.shape == [B, T, N, N, out_dim]

        if self.use_1d is False:
            assert len(inp.shape) == 4
            T, N = out.shape[1], out.shape[2]
            t_emb = self.time_mlp(t)
            t_emb = einops.repeat(t_emb, 'b c -> b t n1 n2 c', t=T, n1=N, n2=N)
            inp = einops.repeat(inp, 'b n1 n2 c -> b t n1 n2 c', t=T)
            x = torch.concat([inp, out, t_emb], dim=-1)
        else:
            assert len(inp.shape) == 3
            T, N = out.shape[1], out.shape[2]
            t_emb = self.time_mlp(t)
            t_emb = einops.repeat(t_emb, 'b c -> b t n1 c', t=T, n1=N)
            inp = einops.repeat(inp, 'b n1 c -> b t n1 c', t=T)
            x = torch.concat([inp, out, t_emb], dim=-1)

        x = self.block1(x)
        x = self.block2(x)
        if self.use_1d is False:
            x = self.block3(x)
        x = self.fc_out(x).squeeze(-1)  # [B, T, N]
        x = x.sum(dim=[1, 2])  # [B]

        return x


class NLMConv1DBlockV2(nn.Module):
    def __init__(self, inp_dim, cond_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.inp_dim * 6 + self.cond_dim * 2, self.out_dim)

    def forward(self, x, cond):
        # x.shape == [B, T, N, inp_dim]
        # cond.shape == [B, T, N, N, cond_dim]

        B, T, N, _ = x.shape
        x1 = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:]], dim=1)
        x2 = torch.cat([x[:, :-1], torch.zeros_like(x[:, -1:])], dim=1)
        x = torch.cat([x1, x, x2], dim=-1)  # [B, T, N, inp_dim * 3]
        x1 = einops.repeat(x, 'b t n1 c -> b t n1 n2 c', n1=N, n2=N)
        x2 = einops.repeat(x, 'b t n2 c -> b t n1 n2 c', n1=N, n2=N)
        cond1 = einops.repeat(cond, 'b n1 n2 c -> b t n1 n2 c', t=T, n1=N, n2=N)
        cond2 = einops.repeat(cond, 'b n1 n2 c -> b t n2 n1 c', t=T, n1=N, n2=N)
        x = torch.cat([x1, x2, cond1, cond2], dim=-1)  # [B, T, N, N, inp_dim * 6 + cond_dim]
        x = self.linear(x)  # [B, T, N, N, out_dim]
        x = swish(x)
        return x.amax(dim=-2)  # [B, T, N, out_dim]


class GNNConv1DEBMV2(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        fourier_dim, time_dim = 512, 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        inp_h = out_dim + time_dim
        cond_h = inp_dim

        self.block1 = NLMConv1DBlockV2(inp_h, cond_h, 64)
        self.block2 = NLMConv1DBlockV2(64, cond_h, 64)
        # self.block3 = NLMConv1DBlockV2(32, cond_h, 32)
        self.fc = nn.Linear(64, 1)

    def forward(self, inp, out, t):
        # inp.shape == [B, N, N, inp_dim]
        # out.shape == [B, T, N, out_dim]

        T, N = out.shape[1], out.shape[2]
        t_emb = self.time_mlp(t)
        t_emb = einops.repeat(t_emb, 'b c -> b t n1 c', t=T, n1=N)
        x = torch.cat([out, t_emb], dim=-1)

        x = self.block1(x, cond=inp)
        x = self.block2(x, cond=inp)
        # x = self.block2(x, cond=inp) + x
        # x = self.block3(x, cond=inp) + x
        x = self.fc(x).squeeze(-1)  # [B, T, N]
        x = x.sum(dim=[1, 2])  # [B]

        return x[:, None]


class GNNConv1DReverse(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        fourier_dim, time_dim = 512, 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        inp_h = out_dim + time_dim
        cond_h = inp_dim

        self.block1 = NLMConv1DBlockV2(inp_h, cond_h, 64)
        self.block2 = NLMConv1DBlockV2(64, cond_h, 64)
        # self.block3 = NLMConv1DBlockV2(32, cond_h, 32)
        self.fc = nn.Linear(64, 1)

    def forward(self, inp, out, t):
        # inp.shape == [B, N, N, inp_dim]
        # out.shape == [B, T, N, out_dim]

        T, N = out.shape[1], out.shape[2]
        t_emb = self.time_mlp(t)
        t_emb = einops.repeat(t_emb, 'b c -> b t n1 c', t=T, n1=N)
        x = torch.cat([out, t_emb], dim=-1)

        x = self.block1(x, cond=inp)
        x = self.block2(x, cond=inp)
        # x = self.block2(x, cond=inp) + x
        # x = self.block3(x, cond=inp) + x
        x = self.fc(x).squeeze(-1)  # [B, T, N]

        return x[:, :, :, None]

    def randn(self, batch_size, shape, inp, device):
        n = inp.shape[-2]
        return torch.randn((batch_size, 8, n, shape[0]), device=device)


class DiffusionWrapper(nn.Module):
    def __init__(self, ebm):
        super(DiffusionWrapper, self).__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

        if hasattr(self.ebm, 'is_ebm'):
            assert self.ebm.is_ebm, 'DiffusionWrapper only works for EBMs'

    def forward(self, inp, opt_out, t, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        opt_variable = torch.cat([inp, opt_out], dim=-1)

        energy = self.ebm(opt_variable, t)

        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]

        if return_both:
            return energy, opt_grad
        else:
            return opt_grad


class GNNDiffusionWrapper(nn.Module):
    def __init__(self, ebm):
        super().__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

    def forward(self, inp, opt_out, t, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        energy = self.ebm(inp, opt_out, t)
        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]
        if return_both:
            return energy, opt_grad
        else:
            return opt_grad

    def randn(self, batch_size, shape, inp, device):
        assert shape[0] == 1
        return torch.randn((batch_size, *inp.shape[1:-1]), device=device)


class GNNConvDiffusionWrapper(nn.Module):
    def __init__(self, ebm):
        super().__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

    def forward(self, inp, opt_out, t, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        energy = self.ebm(inp, opt_out, t)
        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]
        if return_both:
            return energy, opt_grad
        else:
            return opt_grad

    def randn(self, batch_size, shape, inp, device):
        return torch.randn((batch_size, 16, *inp.shape[1:-1], shape[0]), device=device)


class GNNConv1DV2DiffusionWrapper(nn.Module):
    def __init__(self, ebm):
        super().__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

    def forward(self, inp, opt_out, t, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        energy = self.ebm(inp, opt_out, t)
        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]
        if return_both:
            return energy, opt_grad
        else:
            return opt_grad

    def randn(self, batch_size, shape, inp, device):
        n = inp.shape[-2]
        return torch.randn((batch_size, 8, n, shape[0]), device=device)

