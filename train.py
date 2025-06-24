from dataset import create_dataloader
from model_loader import load_model_components
from lora_setup import apply_lora
from utils import setup_training, run_training_loop

def train_model():
    dataloader, prompt_embeddings = create_dataloader()
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_model_components()
    unet = apply_lora(unet)
    trainable_params, optimizer, scheduler, scaler, save_checkpoint = setup_training(unet, dataloader)
    run_training_loop(
        dataloader, tokenizer, text_encoder, vae, unet,
        noise_scheduler, prompt_embeddings,
        trainable_params, optimizer, scheduler, scaler,
        save_checkpoint
    )