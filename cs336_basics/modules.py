from torch import Tensor
from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
import math
from einops import einsum,rearrange

def init_weight(weight):
    in_features, out_features = weight.shape[0], weight.shape[1]
    std = math.sqrt(2.0 / (in_features + out_features))
    torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
    return nn.Parameter(weight)

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device=None, dtype: torch.dtype=None):
        # 初始化权重
        # 使用 truncated normal 初始化
        super().__init__()

        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = init_weight(weight)
    
    def forward(self, x:Float[Tensor, " ... in_features"]):
        # 线性变换
        return einsum(x,self.weight,"... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device=None, dtype:torch.dtype=None):
        # 创建嵌入矩阵
        super().__init__()

        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = init_weight(weight)
    
    def forward(self, token_ids:int):
        # 索引查找
        return self.weight[token_ids]

def Silu(x):
    return x/(torch.ones_like(x) + torch.exp(-x))

class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int=0, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        if d_ff == 0:
            d_ff = int((8/3) * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = Linear(d_ff,d_model)
        self.w2 = Linear(d_model,d_ff)
        self.w3 = Linear(d_ff,d_model)
    
    def forward(self,x):
        gate = self.w1(x)
        value = self.w3(x)
        gate_activated = Silu(gate)
        output = self.w2(gate_activated*value)
        return output

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einsum(Q ,K,"... queries d_k,  ... keys d_k -> ... queries keys")/math.sqrt(d_k)

    if mask is not None:
        # mask 中为 False 的位置设为 -inf
        scores = scores.masked_fill(~mask, float('-inf'))

    attention_weights = softmax(scores,-1)

    return einsum(attention_weights,V,"... queries keys, ... keys d_v -> ... queries d_v")

def softmax(input:Tensor,dim:int) -> torch.Tensor:
    max_val, _ = torch.max(input,dim,keepdim=True)
    shifted = input - max_val
    e_x = torch.exp(shifted)
    return e_x/torch.sum(e_x,dim,keepdim=True)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model//num_heads
        self.Q = Linear(d_model,d_model)
        self.K = Linear(d_model,d_model)
        self.V = Linear(d_model,d_model)
        self.O = Linear(d_model,d_model)
        self.rope = RoPE(self.d_h,max_seq_len,theta)

    def forward(self, x: Float[Tensor, " ... sequence_length d_in"], positions: Int[Tensor, " ... sequence_length"] | None = None ,use_rope: bool = False) -> Float[Tensor, " ... sequence_length d_out"]:
        seq_len = x.shape[-2]
        device = x.device
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        Q = rearrange(Q,"... sequence_length (num_heads d_h) -> ... num_heads sequence_length d_h", num_heads=self.num_heads)
        K = rearrange(K,"... sequence_length (num_heads d_h) -> ... num_heads sequence_length d_h", num_heads=self.num_heads)
        if use_rope == True:
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)
        V = rearrange(V,"... sequence_length (num_heads d_h) -> ... num_heads sequence_length d_h", num_heads=self.num_heads)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0).bool()
        output = scaled_dot_product_attention(Q,K,V,causal_mask)
        output = rearrange(output,"... num_heads sequence_length d_h -> ... sequence_length (num_heads d_h)")
        output = self.O(output)
        return output

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs = rearrange(freqs, "half_dim -> 1 half_dim")
        
        # 预计算所有位置的 cos/sin
        positions = rearrange(torch.arange(max_seq_len, dtype=torch.float32), "max_seq_len -> max_seq_len 1")
        angles = einsum(positions, freqs, "max_seq_len one,one half_dim -> max_seq_len half_dim")
        
        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, positions: Int[Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        """
        x: (..., seq_len, dim)
        positions: (..., seq_len) or None
        """
        if positions is None:
            # 默认位置 0, 1, 2, ...
            cos = self.cos[:x.shape[-2]]
            sin = self.sin[:x.shape[-2]]
        else:
            cos = self.cos[positions]
            sin = self.sin[positions]
        
        # 拆分并旋转
        x_even = x[..., 0::2]  # (..., seq_len, half_dim)
        x_odd = x[..., 1::2]   # (..., seq_len, half_dim)
        
        # 旋转（需要广播）
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        # 合并
        rotated = torch.zeros_like(x)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        
        return rotated
        
        





class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        # 初始化 gain 参数
        pass
    
    def forward(self, x):
        # 实现 RMSNorm
        # 注意：需要 upcast 到 float32
        pass

