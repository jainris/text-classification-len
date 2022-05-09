import math

import torch
from torch import Tensor
from torch import nn

from .concepts import Conceptizator


def broadcast_sparse_dense_elemwise_mul(
    sparse_tensor, dense_tensor, requires_grad=False
):
    device = sparse_tensor.device

    def get_broadcasted_shape(s1, s2):
        new_shape = []

        for i in range(len(s1)):
            if s1[i] != 1:
                new_shape.append(s1[i])
            else:
                new_shape.append(s2[i])

        return new_shape

    def get_num_repeats(size_st, new_shape):
        num_repeats = 1
        for i in range(len(size_st)):
            if size_st[i] != new_shape[i]:
                num_repeats *= new_shape[i]
        return num_repeats

    def update_values_and_indices(sparse_tensor, reps, size_st, new_shape):
        idx = sparse_tensor._indices()
        v = sparse_tensor._values()

        new_idx = []

        v = v.repeat(reps)
        n = v.size(0)
        for i in range(len(size_st)):
            if size_st[i] == new_shape[i]:
                new_idx.append(idx[i].repeat(reps))
            else:
                new_idx.append(
                    torch.arange(new_shape[i], device=device).repeat(n // new_shape[i])
                )

        return v, new_idx

    size_st = sparse_tensor.size()
    size_dt = dense_tensor.size()

    new_shape = get_broadcasted_shape(size_st, size_dt)
    reps = get_num_repeats(size_st, new_shape)
    v, i = update_values_and_indices(sparse_tensor, reps, size_st, new_shape)

    dense_tensor = dense_tensor.expand(new_shape)
    v = dense_tensor[i] * v
    i = torch.vstack(i)
    return torch.sparse.FloatTensor(i, v, new_shape).requires_grad_(requires_grad)


class SparseDenseElemMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_tensor: torch.Tensor, dense_tensor: torch.Tensor):
        ctx.save_for_backward(sparse_tensor, dense_tensor)
        requires_grad = sparse_tensor.requires_grad or dense_tensor.requires_grad
        output = broadcast_sparse_dense_elemwise_mul(
            sparse_tensor, dense_tensor, requires_grad=requires_grad
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparse_tensor, dense_tensor = ctx.saved_tensors
        grad_sparse_tensor = None
        grad_dense_tensor = None
        if ctx.needs_input_grad[0]:
            grad_sparse_tensor = broadcast_sparse_dense_elemwise_mul(
                grad_output, dense_tensor
            )

            size_g = grad_sparse_tensor.size()
            size_st = sparse_tensor.size()
            dims = []
            for i in range(len(size_st)):
                if size_st[i] != size_g[i]:
                    dims.append(i)

            grad_sparse_tensor = torch.sparse.sum(grad_sparse_tensor, dim=dims)
            for d in dims:
                grad_sparse_tensor = grad_sparse_tensor.unsqueeze(d)
        if ctx.needs_input_grad[1]:
            # grad_dense_tensor = sparse_sparse_mul(grad_output, sparse_tensor)
            grad_dense_tensor = None  # TODO: Implement this

        return grad_sparse_tensor, grad_dense_tensor


sdem = SparseDenseElemMul.apply


def batch_sparse_matmul(sparse_tensor, dense_tensor):
    tens = []
    for i in range(sparse_tensor.size(0)):
        tens.append(torch.sparse.mm(sparse_tensor[i], dense_tensor[i]).unsqueeze(0))
    return torch.vstack(tens)


class EntropyLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_classes: int,
        temperature: float = 0.6,
        bias: bool = True,
        conceptizator: str = "identity_bool",
        sparse : bool = False,
    ) -> None:
        super(EntropyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.conceptizator = Conceptizator(conceptizator)
        self.alpha = None
        self.weight = nn.Parameter(torch.Tensor(n_classes, out_features, in_features))
        self.sparse = sparse
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_classes, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        # compute concept-awareness scores
        gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(gamma / self.temperature) / torch.sum(
            torch.exp(gamma / self.temperature), dim=1, keepdim=True
        )

        # weight the input concepts by awareness scores
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = self.alpha_norm > 0.5
        if self.sparse:
            x = sdem(input, self.alpha_norm.unsqueeze(1))
        else:
            x = input.multiply(self.alpha_norm.unsqueeze(1))

        # compute linear map
        if self.sparse:
            x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        else:
            x = batch_sparse_matmul(x, self.weight.permute(0, 2, 1)) + self.bias
        return x.permute(1, 0, 2)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_classes={}".format(
            self.in_features, self.out_features, self.n_classes
        )


if __name__ == "__main__":
    data = torch.rand((10, 5))
    layer = EntropyLinear(5, 4, 2)
    out = layer(data)
    print(out.shape)
