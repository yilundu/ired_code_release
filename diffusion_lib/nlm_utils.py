from enum import Enum
from typing import Union, Tuple, List
import collections
import itertools
import torch
import torch.nn as nn


def broadcast(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Broadcast a specific dim for `size` times. Originally the dim size must be 1.

    Example:

        >>> broadcast(torch.tensor([1, 2, 3]), 0, 2)
        tensor([[1, 2, 3],
                [1, 2, 3]])

    Args:
        tensor: the tensor to be broadcasted.
        dim: the dimension to be broadcasted.
        size: the size of the target dimension.

    Returns:
        the broadcasted tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim + 1:]))


def concat_shape(*shapes: Union[torch.Size, Tuple[int, ...], List[int], int]) -> Tuple[int, ...]:
    """Concatenate shapes into a tuple. The values can be either torch.Size, tuple, list, or int."""
    output = []
    for s in shapes:
        if isinstance(s, collections.abc.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def meshgrid(input1, input2=None, dim=-1):
    """Perform np.meshgrid along given axis. It will generate a new dimension after dim."""
    if input2 is None:
        input2 = input1
    if dim < 0:
        dim += input1.dim()
    n, m = input1.size(dim), input2.size(dim)
    x = broadcast(input1.unsqueeze(dim + 1), dim + 1, m)
    y = broadcast(input2.unsqueeze(dim + 0), dim + 0, n)
    return x, y


def meshgrid_exclude_self(input, dim=1):
    """
    Exclude self from the grid. Specifically, given an array a[i, j] of n * n, it produces
    a new array with size n * (n - 1) where only a[i, j] (i != j) is preserved.

    The operation is performed over dim and dim +1 axes.
    """
    if dim < 0:
        dim += input.dim()

    n = input.size(dim)
    assert n == input.size(dim + 1)

    # exclude self-attention
    rng = torch.arange(0, n, dtype=torch.long, device=input.device)
    rng_n1 = rng.unsqueeze(1).expand((n, n))
    rng_1n = rng.unsqueeze(0).expand((n, n))
    mask_self = (rng_n1 != rng_1n)

    for i in range(dim):
        mask_self.unsqueeze_(0)
    for j in range(input.dim() - dim - 2):
        mask_self.unsqueeze_(-1)
    target_shape = concat_shape(input.size()[:dim], n, n-1, input.size()[dim+2:])

    return input.masked_select(mask_self).view(target_shape)


def exclude_mask(input, cnt=2, dim=1):
    """
    Produce exclude mask. Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

    The operation is performed over [dim, dim + cnt) axes.
    """
    assert cnt > 0
    if dim < 0:
        dim += input.dim()
    n = input.size(dim)
    for i in range(1, cnt):
        assert n == input.size(dim + i)

    rng = torch.arange(0, n, dtype=torch.long, device=input.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(input.dim() - dim - cnt):
        mask.unsqueeze_(-1)

    return mask.expand(input.size()).float()


def mask_value(input, mask, value):
    assert input.size() == mask.size()
    return input * mask + value * (1 - mask)


# Capture a free variable into predicates, implemented by broadcast
class Expander(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input, n=None):
        if self.dim == 0:
            assert n is not None
        elif n is None:
            n = input.size(self.dim)
        dim = self.dim + 1
        return broadcast(input.unsqueeze(dim), dim, n)

    def get_output_dim(self, input_dim):
        return input_dim


# Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling
class Reducer(nn.Module):
    def __init__(self, dim, exclude_self=True, exists=True):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.exists = exists

    def forward(self, input):
        shape = input.size()
        inp0, inp1 = input, input
        if self.exclude_self:
            mask = exclude_mask(input, cnt=self.dim, dim=-1 - self.dim)
            inp0 = mask_value(input, mask, 0.0)
            inp1 = mask_value(input, mask, 1.0)

        if self.exists:
            shape = shape[:-2] + (shape[-1] * 2, )
            exists = torch.max(inp0, dim=-2)[0]
            forall = torch.min(inp1, dim=-2)[0]
            return torch.stack((exists, forall), dim=-1).view(shape)

        shape = shape[:-2] + (shape[-1], )
        return torch.max(inp0, dim=-2)[0].view(shape)

    def get_output_dim(self, input_dim):
        if self.exists:
            return input_dim * 2
        return input_dim


class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        if self.dim <= 1:
            return input
        nr_dims = len(input.size())
        # Assume the last dim is channel.
        index = tuple(range(nr_dims - 1))
        start_dim = nr_dims - 1 - self.dim
        assert start_dim > 0
        res = []
        for i in itertools.permutations(index[start_dim:]):
            p = index[:start_dim] + i + (nr_dims - 1,)
            res.append(input.permute(p))
        return torch.cat(res, dim=-1)

    def get_output_dim(self, input_dim):
        mul = 1
        for i in range(self.dim):
            mul *= i + 1
        return input_dim * mul


def get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    elif bn == 'sync':
        raise NotImplementedError()
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))


def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'identity':
        return nn.Identity()
    elif act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

        self.in_features = in_features
        self.out_features = out_features

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True, last_activation=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation
        self.flatten = flatten
        self.last_activation = last_activation

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=self.batch_norm, dropout=self.dropout, activation=self.activation)
            modules.append(layer)
        if self.last_activation:
            layer = LinearLayer(dims[-2], dims[-1], batch_norm=self.batch_norm, dropout=self.dropout, activation=self.activation)
        else:
            layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)


class InferenceBase(nn.Module):
    """MLP model with shared parameters among other axies except the channel axis."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer = nn.Sequential(MLPLayer(input_dim, output_dim, hidden_dim))

    def forward(self, inputs):
        input_size = inputs.size()[:-1]
        input_channel = inputs.size(-1)

        f = inputs.view(-1, input_channel)
        f = self.layer(f)
        f = f.view(*input_size, -1)
        return f


def get_output_dim(self, input_dim):
    return self.output_dim


class LogicInference(InferenceBase):
    """MLP layer with sigmoid activation."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__(input_dim, output_dim, hidden_dim)
        self.layer.add_module(str(len(self.layer)), nn.Sigmoid())


class LogitsInference(InferenceBase):
    pass


class InputTransformMethod(Enum):
    CONCAT = 'concat'
    DIFF = 'diff'
    CMP = 'cmp'


class InputTransform(nn.Module):
    """Transform the unary predicates to binary predicates by operations."""

    def __init__(self, method, exclude_self=True):
        super().__init__()
        self.method = InputTransformMethod.from_string(method)
        self.exclude_self = exclude_self

    def forward(self, inputs):
        assert inputs.dim() == 3

        x, y = meshgrid(inputs, dim=1)

        if self.method is InputTransformMethod.CONCAT:
            combined = torch.cat((x, y), dim=3)
        elif self.method is InputTransformMethod.DIFF:
            combined = x - y
        elif self.method is InputTransformMethod.CMP:
            combined = torch.cat([x < y, x == y, x > y], dim=3)
        else:
            raise ValueError('Unknown input transform method: {}.'.format(self.method))

        if self.exclude_self:
            combined = meshgrid_exclude_self(combined, dim=1)
        return combined.float()

    def get_output_dim(self, input_dim):
        if self.method is InputTransformMethod.CONCAT:
            return input_dim * 2
        elif self.method is InputTransformMethod.DIFF:
            return input_dim
        elif self.method is InputTransformMethod.CMP:
            return input_dim * 3
        else:
            raise ValueError('Unknown input transform method: {}.'.format(self.method))

    def __repr__(self):
        return '{name}({method}, exclude_self={exclude_self})'.format(name=self.__class__.__name__, **self.__dict__)

