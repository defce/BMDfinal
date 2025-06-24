from peft import LoraConfig, get_peft_model

def apply_lora(unet):
    config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none"
    )
    return get_peft_model(unet, config)