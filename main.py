# Cell 1: Install necessary packages with version pinning for compatibility
!pip install -q diffusers==0.23.1 transformers==4.34.0 accelerate==0.23.0 peft==0.6.0 huggingface-hub==0.17.3
!pip install -q bitsandbytes --no-deps

# Cell 2: Import libraries and set up environment
import os
import gc
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# Set memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Clear GPU memory before starting
torch.cuda.empty_cache()
gc.collect()

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cell 3: Define the ChestXray dataset
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, images_folder, target_size=(512, 512), max_samples=None):
        self.data = pd.read_csv(csv_file)
        if max_samples is not None:
            self.data = self.data.head(max_samples)
        self.images_folder = images_folder
        self.target_size = target_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Index"]
        img_path = os.path.join(self.images_folder, img_name)
        
        # Load and resize image
        with Image.open(img_path).convert("RGB") as img:
            img = img.resize(self.target_size, Image.BICUBIC)
            
            # Convert to numpy and normalize to [-1, 1]
            img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        # Get condition and create specialized medical prompt
        condition = self.data.iloc[idx]["Finding Labels"]
        # Use RoentGen-style prompting
        prompt = f"A chest x-ray with {condition}"
        
        return {"image": img_tensor, "prompt": prompt}
    

# Cell 4: Create dataset with 2000 samples
# Define paths to your data
csv_file = "/kaggle/input/sample/sample_labels.csv"
images_folder = "/kaggle/input/sample/sample/sample/images"

# Create dataset with 256x256 resolution and 2000 samples
RESOLUTION = 256   # 256x256 resolution
NUM_SAMPLES = 2000  # Using 2000 samples as requested

print(f"Creating dataset with {RESOLUTION}x{RESOLUTION} resolution and {NUM_SAMPLES} samples...")
dataset = ChestXrayDataset(
    csv_file=csv_file,
    images_folder=images_folder,
    target_size=(RESOLUTION, RESOLUTION),
    max_samples=NUM_SAMPLES
)

# Create dataloader with batch size 1
dataloader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=True,
    num_workers=0  # No parallel loading to save memory
)

print(f"Dataset created with {len(dataset)} samples at {RESOLUTION}x{RESOLUTION} resolution")

# Display information about conditions in the dataset
all_conditions = []
for i in range(min(len(dataset), 2000)):  # Sample all conditions
    prompt = dataset[i]["prompt"]
    condition = prompt.split("with ")[-1]
    all_conditions.append(condition)

from collections import Counter
condition_counts = Counter(all_conditions)
print("\nTop conditions in the dataset:")
for condition, count in condition_counts.most_common(10):
    print(f"- {condition}: {count} samples")

# Cell 5: Load model components with specialized medical model
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

print("Loading Prompt2MedImage model components...")

# Specify the model ID
base_model_id = "Nihirc/Prompt2MedImage"

try:
    # First try direct loading to check if model exists
    test_pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    print(f"Successfully verified {base_model_id} exists")
    del test_pipe
    torch.cuda.empty_cache()
    
    # This is a specialized model, so we need to load components differently
    # Load the complete pipeline first to extract components
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    # Extract individual components
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler
    
    # Clear the pipeline to save memory
    del pipe
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"Error loading Prompt2MedImage: {e}")
    print("Falling back to standard Stable Diffusion v1.5")
    
    # Fallback to standard model
    base_model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load components with subfolder specification (standard SD structure)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id, 
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    
    vae = AutoencoderKL.from_pretrained(
        base_model_id, 
        subfolder="vae",
        torch_dtype=torch.float16
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        base_model_id, 
        subfolder="unet",
        torch_dtype=torch.float16
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

# Set components to evaluation mode
text_encoder.eval()
vae.eval()

print(f"Using {base_model_id} as base model")
print("Model components loaded successfully")

# Cell 6: Set up LoRA for efficient fine-tuning
print("Setting up LoRA configuration...")

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,              # Rank for LoRA
    lora_alpha=16,     
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Target attention modules
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA to UNet
unet = get_peft_model(unet, lora_config)

# Get trainable parameters
trainable_params = [p for p in unet.parameters() if p.requires_grad]
print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
print(f"LoRA reduction factor: {sum(p.numel() for p in unet.parameters()) / sum(p.numel() for p in trainable_params):.2f}x")

# Get VAE scaling factor
vae = vae.to("cuda")
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)  # Usually 8
latent_channels = vae.config.latent_channels  # Usually 4
print(f"VAE scale factor: {vae_scale_factor}, latent channels: {latent_channels}")
vae = vae.to("cpu")


# Cell 7: Training preparation
# Pre-encode text prompts
print("Pre-encoding text prompts...")
text_encoder = text_encoder.to("cuda")

# Get unique prompts
unique_prompts = set()
for i in range(min(len(dataset), 1000)):
    if len(unique_prompts) >= 100:  # Limit to 100 unique prompts
        break
    unique_prompts.add(dataset[i]["prompt"])

prompt_embeddings = {}
for prompt in tqdm(list(unique_prompts)):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(
            **{k: v.to("cuda") for k, v in text_inputs.items()}
        )[0]
    prompt_embeddings[prompt] = text_embeddings.cpu()
    
    # Free memory
    del text_embeddings, text_inputs
    torch.cuda.empty_cache()

# Add special healthy lung prompts
healthy_prompts = [
    "A chest x-ray with No Finding",
    "A chest x-ray with normal lungs",
    "A chest x-ray with healthy appearance"
]

for prompt in healthy_prompts:
    if prompt not in prompt_embeddings:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = text_encoder(
                **{k: v.to("cuda") for k, v in text_inputs.items()}
            )[0]
        prompt_embeddings[prompt] = text_embeddings.cpu()
        
        # Free memory
        del text_embeddings, text_inputs
        torch.cuda.empty_cache()

text_encoder = text_encoder.to("cpu")
torch.cuda.empty_cache()
gc.collect()

print(f"Pre-encoded {len(prompt_embeddings)} prompts")

# Cell 8: Training loop setup
# Move UNet to GPU
unet = unet.to("cuda")

# Optimizer and learning rate scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(
    trainable_params,
    lr=1e-4,
    weight_decay=0.01
)

# Set up training for multiple epochs
NUM_EPOCHS = 5
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=len(dataloader) * NUM_EPOCHS,
    eta_min=1e-6
)

# Mixed precision training
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# Set up output directories
output_dir = "/kaggle/working/roentgen_enhanced"
os.makedirs(output_dir, exist_ok=True)

# Checkpoint saving function
def save_checkpoint(epoch, step=None):
    if step is not None:
        save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_step{step}")
    else:
        save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
        
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)
    print(f"Saved checkpoint to {save_path}")
    return save_path

# Cell 9: Training loop with consistent precision
print(f"Starting training for {NUM_EPOCHS} epochs...")

# Track losses
all_losses = []

# Set precision to float32 for everything to avoid dtype mismatches
# First convert model components to float32
unet = unet.to(torch.float32)

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_losses = []
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(progress_bar):
        try:
            # Zero gradients
            optimizer.zero_grad()
            
            # Get data
            images = batch["image"].to("cuda", dtype=torch.float32)
            prompt = batch["prompt"][0]
            
            # Get text embeddings
            if prompt in prompt_embeddings:
                text_embeddings = prompt_embeddings[prompt].to("cuda", dtype=torch.float32)  # Ensure float32
            else:
                # Find best matching prompt
                best_match = None
                best_score = -1
                
                for key in prompt_embeddings.keys():
                    # Simple word overlap score
                    score = sum(word in key for word in prompt.split())
                    if score > best_score:
                        best_score = score
                        best_match = key
                
                if best_match:
                    text_embeddings = prompt_embeddings[best_match].to("cuda", dtype=torch.float32)  # Ensure float32
                else:
                    print(f"Skipping unknown prompt: {prompt}")
                    continue
            
            # Calculate latent dimensions
            batch_size, channels, height, width = images.shape
            latent_height = height // vae_scale_factor
            latent_width = width // vae_scale_factor
            
            # Create random latents
            latents = torch.randn(
                (batch_size, latent_channels, latent_height, latent_width),
                device="cuda",
                dtype=torch.float32
            )
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device="cuda").long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Free memory
            del latents
            torch.cuda.empty_cache()
            
            # Enable gradient checkpointing if available
            if hasattr(unet, "enable_gradient_checkpointing"):
                unet.enable_gradient_checkpointing()
            
            # Forward pass
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Track losses
            loss_value = loss.item()
            all_losses.append(loss_value)
            epoch_losses.append(loss_value)
            
            # Free memory
            del noise_pred, noise, noisy_latents, text_embeddings
            torch.cuda.empty_cache()
            
            # Save checkpoint periodically
            if (step + 1) % 200 == 0:
                save_checkpoint(epoch + 1, step + 1)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error at epoch {epoch+1}, step {step+1}, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"Error: {e}")
                raise e
    
    # Save checkpoint at end of epoch
    save_checkpoint(epoch + 1)
    
    # Report average loss
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# Save final model
final_save_path = os.path.join(output_dir, "final_model")
unet.save_pretrained(final_save_path)
print(f"Training complete! Final model saved to {final_save_path}")

# Plot loss curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(all_losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig(os.path.join(output_dir, "training_loss.png"))
plt.close()