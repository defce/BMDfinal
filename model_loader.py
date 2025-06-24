import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

def load_model_components():
    base_model_id = "Nihirc/Prompt2MedImage"
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    return pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet, pipe.scheduler