import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0., qkv_bias=True, qk_scale=None):
        super().__init__()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.n_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, C//self.n_heads).permute(2, 0, 3, 1, 4) # `3, B_, num_heads, N, head_dims`
        q, k, v = qkv[0], qkv[1], qkv[2] # `B_, num_heads, N, head_dims`

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0., mlp_drop=0., attn_drop=0.):
        super().__init__()

        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadedSelfAttention(dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=mlp_drop)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1.norm 2.multi-head self-attention 3.short cut 4.norm 5.mlp 6.short cut
        shortcut = x
        x = self.norm1(x)
        atten = self.attn(x, mask)
        x = shortcut + self.drop( self.proj(atten) )

        h = self.drop( self.mlp( self.norm2(x) ) )
        x = x + h

        return x


class TransformerEncoder(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, dim, num_heads, num_layers,
                 mlp_ratio=4., dropout=0., mlp_drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, dropout, mlp_drop, attn_drop) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

def build_transformer(args):
    return TransformerEncoder(
        num_layers = args.num_layers,
        dim = args.dim,
        num_heads = args.n_heads
    )