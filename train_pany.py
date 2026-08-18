import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import argparse
import random
import warnings
from hydra import initialize, compose
from training.lightning_pany import PANY
from training.data.datasets.ycbv import YCBVDataset
from training.data.datasets.omni6d import Omni6DDataModule, Omni6DDataset

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
        if ("lora_" in param_name) or param_name.startswith("depth_head") or param_name.startswith("point_head") or param_name.startswith("track_head"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model
    
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="pany_model",
        help="Name of the config file (without .yaml extension, default: pany_model)"
    )
    args = parser.parse_args()
    with initialize(version_base=None, config_path="training"):
        config = compose(config_name=args.config)
    
    # Initialize the PANY model
    model = PANY(config)
    if config.per_trained:
        # Load the pre-trained model
        ckpt_path = "ckpt/model.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model = lora_to_global_attention(model)

    data_root = '/media/tum/data1/Omni6DPose/data/Omni6DPose/SOPE'
    model_meta_root = '/media/tum/data1/Omni6DPose/data/Omni6DPose/Meta'
    
    Omni6D_train_dataset = Omni6DDataset(common_conf=config.common_config,
                                  data_root=data_root,
                                  model_meta_root=model_meta_root)
    Omni6D_val_dataset = YCBVDataset(common_conf=config.common_config)
    data_loader = Omni6DDataModule(train_dataset=Omni6D_train_dataset,
                                val_dataset=Omni6D_val_dataset, 
                                batch_size=config.common_config.batch_size, 
                                num_workers=4, 
                                train_steps_per_epoch=config.train_steps_per_epoch,
                                val_steps_per_epoch=config.val_steps_per_epoch)
    
    # Initial Wandb
    wandb_logger = WandbLogger(
        project="PANY",        # your project name
        name="run_name",            # optional: name of this specific run
        log_model=False             # or True if you want to log model checkpoints
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",           
        mode="min",                      
        save_top_k=1,                     
        save_last=True,
        save_weights_only=True,                
        dirpath="ckpt/",
        every_n_train_steps=125,                   
        filename="best-{epoch:02d}-{train_loss:.4f}",  
    )
    
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=1, 
                         max_steps=config.total_steps,    # total training steps
                         accumulate_grad_batches=16,
                         precision="bf16-mixed", 
                         reload_dataloaders_every_n_epochs=0,
                         limit_train_batches=config.train_steps_per_epoch,  
                         limit_val_batches=config.val_steps_per_epoch,     
                         check_val_every_n_epoch=1,
                         profiler="simple",
                         callbacks=[checkpoint_callback],
                         strategy=DDPStrategy(find_unused_parameters=True),
                        #  logger=wandb_logger,  # Disable default logger
                         )
    trainer.fit(model, data_loader)





