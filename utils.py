import os, torch, gc
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

def setup_training(unet, dataloader):
    unet = unet.to("cuda")
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader)*5, eta_min=1e-6)
    scaler = GradScaler()
    def save_checkpoint(epoch):
        path = f"checkpoints/epoch_{epoch}"
        os.makedirs(path, exist_ok=True)
        unet.save_pretrained(path)
    return params, optimizer, scheduler, scaler, save_checkpoint

def run_training_loop(*args):
    print("[Training loop stub] Replace with real logic from original training code.")