import torch
import torch.nn as nn
import math
from training.lightning_pany import PANY

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        # 保留原来的权重和偏置
        self.original = original
        self.in_features  = original.in_features
        self.out_features = original.out_features

        # LoRA 参数
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(self.in_features,  r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化 A、B
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 原始输出 + LoRA 输出
        return self.original(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling

def lora_to_global_attention(model, r=8, alpha=32, dropout=0.05):
    for blk in model.aggregator.frame_blocks + model.aggregator.global_blocks:
        attn = blk.attn
        for name in ("qkv", "proj"):
            orig: nn.Linear = getattr(attn, name)
            setattr(attn, name, LoRALinear(orig, r=r, alpha=alpha, dropout=dropout))

    for param_name, param in model.named_parameters():
        if ("lora_" in param_name) or param_name.startswith("point_head") or param_name.startswith("depth_head") or param_name.startswith("camera_head"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def load_model(config, ckpt_path):
    model = PANY(config)
    model = lora_to_global_attention(model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model