import torch.nn as nn
import torch


class SelfAttention(nn.Module):

    def __init__(
            self, d_in: int, d_out: int,
            context_length: int, dropout: float, qkv_bias=False):
        super(SelfAttention, self,).__init__()
        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        att_scores = q @ k.transpose(1, 2)
        att_scores = att_scores.masked_fill(
            self.mask.bool()[: num_tokens, : num_tokens], -torch.inf)
        att_weights = torch.softmax(att_scores / k.shape[-1]**0.5, dim=-1)
        att_weights = self.dropout(att_weights)
        ctx_vec = att_weights @ v
        return ctx_vec


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(
            self, d_in: int, d_out: int,
            context_length: int, dropout: float,
            num_heads: int, qkv_bias=False):
        super(MultiHeadAttentionWrapper, self).__init__()
        self.heads = nn.ModuleList([
            SelfAttention(
                d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):

    def __init__(
            self, d_in: int, d_out: int,
            context_length: int, dropout: float,
            num_heads: int, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att_scores = q @ k.transpose(2, 3)
        att_scores = att_scores.masked_fill(
            self.mask.bool()[: num_tokens, : num_tokens], -torch.inf)
        att_weights = torch.softmax(att_scores / k.shape[-1]**0.5, dim=-1)
        att_weights = self.dropout(att_weights)
        ctx_vec = (att_weights @ v).transpose(1, 2)
        ctx_vec = ctx_vec.contiguous().view(b, num_tokens, self.d_out)  # why do we need contiguous?
        return self.out_proj(ctx_vec)


torch.manual_seed(123)
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55]
    ]  # step     (x^6)
)
batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1]
d_out = 2
# self_attention = SelfAttention(
#     inputs.shape[1], 2, context_length=context_length, dropout=0.0)
# res = self_attention(batch)

# mha = MultiHeadAttentionWrapper(
#     inputs.shape[1], d_out, context_length, 0.0, num_heads=2)
# res = mha(batch)

mha = MultiHeadAttention(
    inputs.shape[1], 2, context_length, 0.0, num_heads=2)
res = mha(batch)

print(res)
