# Cell 10: Generate images for evaluation
print("Generating test images...")

# Import PeftModel
from peft import PeftModel

# Load the pipeline with our trained weights
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16
)

# Disable safety checker for medical images
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# Load our LoRA weights
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    final_save_path,
    adapter_name="default"
)

# Move to GPU
pipe = pipe.to("cuda")

# Test with various healthy lung prompts
test_prompts = [
    "A chest x-ray with No Finding",
    "A chest x-ray with normal lungs",
    "A chest x-ray with healthy appearance",
    "A chest x-ray with clear lung fields",
    "A chest x-ray with no abnormalities"
]

test_output_dir = os.path.join(output_dir, "test_images")
os.makedirs(test_output_dir, exist_ok=True)

# Generate multiple versions with different seeds
for i, prompt in enumerate(test_prompts):
    for seed in [42, 123, 456, 789, 1024]:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"Generating: {prompt} (seed={seed})")
        
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512
        ).images[0]
        
        # Save image
        filename = f"healthy_{i+1}_seed{seed}.png"
        image.save(os.path.join(test_output_dir, filename))
        
        # Clear cache
        torch.cuda.empty_cache()

# Display test images
plt.figure(figsize=(15, 10))
images = [f for f in os.listdir(test_output_dir) if f.endswith('.png')]
for i, img_name in enumerate(images[:12]):
    img = Image.open(os.path.join(test_output_dir, img_name))
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"Prompt {img_name.split('_')[1]}, Seed {img_name.split('seed')[1].split('.')[0]}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(test_output_dir, "results_grid.png"))
plt.show()