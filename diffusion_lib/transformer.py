import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, hidden_dim: int, nr_heads: int, max_length: int, attention_dropout_p: float = 0.1, residual_dropout_p: float = 0.1, use_causal_mask: bool = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.nr_heads = nr_heads
        self.max_length = max_length
        self.attention_dropout_p = attention_dropout_p
        self.residual_dropout_p = residual_dropout_p
        self.use_causal_mask = use_causal_mask

        assert hidden_dim % nr_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        # regularization
        self.attention_dropout = nn.Dropout(attention_dropout_p)
        self.residual_dropout = nn.Dropout(residual_dropout_p)
        # output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length),
        )

    def forward(self, x, layer_past=None):
        if isinstance(x, tuple):  # the output could be (feat, attention_map)
            x = x[0]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.nr_heads, C // self.nr_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.nr_heads, C // self.nr_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.nr_heads, C // self.nr_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.use_causal_mask:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        att = self.attention_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.residual_dropout(self.proj(y))
        return y, att_to_check


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, hidden_dim: int, nr_heads: int, max_length: int, attention_dropout_p: float = 0.1, residual_dropout_p: float = 0.1, use_causal_mask: bool = False, use_time_mapping: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, nr_heads, max_length, attention_dropout_p, residual_dropout_p, use_causal_mask)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout_p),
        )
        self.use_time_mapping = use_time_mapping
        if self.use_time_mapping:
            self.time_map = nn.Linear(hidden_dim, 2 * hidden_dim)
        else:
            self.add_module('time_map', None)

    def forward(self, x, time_embedding=None):
        if isinstance(x, tuple):  # the output could be (feat, attention_map)
            x = x[0]
        # x = x + self.attn(self.ln1(x))
        att, att_to_check = self.attn(self.ln1(x))
        x = x + att
        residual = self.mlp(self.ln2(x))

        if self.use_time_mapping:
            assert time_embedding is not None
            time_map = self.time_map(time_embedding)
            time_gain, time_bias = time_map.chunk(2, dim=-1)
            residual = residual * (time_gain.unsqueeze(1) + 1) + time_bias.unsqueeze(1)

        x = x + residual
        return x, att_to_check


class GPT(nn.Module):
    def __init__(self, nr_layers: int, hidden_dim: int, nr_heads: int, max_length: int, output_dim: int, embedding_dropout_p: float = 0.1, attention_dropout_p: float = 0.1, residual_dropout_p: float = 0.1, use_causal_mask: bool = False):
        super().__init__()

        self.nr_layers = nr_layers
        self.hidden_dim = hidden_dim
        self.nr_heads = nr_heads
        self.max_length = max_length
        self.output_dim = output_dim
        self.embedding_dropout_p = embedding_dropout_p
        self.attention_dropout_p = attention_dropout_p
        self.residual_dropout_p = residual_dropout_p
        self.use_causal_mask = use_causal_mask

        # encoder
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout_p)
        # transformer
        self.blocks = nn.Sequential(*[Block(hidden_dim, nr_heads, max_length, attention_dropout_p, residual_dropout_p, use_causal_mask) for _ in range(nr_layers)])
        # decoder head

        if self.output_dim > 0:
            self.ln_final = nn.LayerNorm(hidden_dim)
            self.time_map = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.fc_final = nn.Linear(hidden_dim, hidden_dim)
            self.head_final = nn.Linear(hidden_dim, output_dim)
        else:
            self.add_module('ln_final1', None)
            self.add_module('time_map', None)
            self.add_module('fc_final', None)
            self.add_module('head_final', None)

        self.apply(self._init_weights)

        print("number of parameters: %e" % sum(p.numel() for p in self.parameters()))
        print("number of trainable parameters: %e" % sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def make_optimizer(self, lr, betas, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('positional_embedding')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def forward(self, input_tensor, time_embedding=None):
        """
        Returns:
            the loss as a scalar
            the logits in the final prediction; (batch_size, 81, 9)
            the attention for the 1st data in a batch; (n_layer * n_recur, num_heads, 81, 81)
        """
        _, t = input_tensor.shape[0], input_tensor.shape[1]
        assert t <= self.positional_embedding.shape[1], "Cannot forward, model block size is exhausted."
        assert input_tensor.shape[2] == self.hidden_dim

        # forward the GPT model
        token_embeddings = input_tensor
        position_embeddings = self.positional_embedding[:, :t, :]  # each position maps to a (learnable) vector
        x = self.embedding_dropout(token_embeddings + position_embeddings)

        atts = []
        for block in self.blocks:
            x, att_to_check = block(x)  # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
            atts.append(att_to_check)

        if self.output_dim > 0:
            h = self.fc_final(self.ln_final(x))
            fc_gain, fc_bias = torch.chunk(self.time_map(time_embedding), 2, dim=-1)
            h = F.gelu(h * (fc_gain.unsqueeze(1) + 1) + fc_bias.unsqueeze(1))
            x = self.head_final(h)
            return x
        return x

