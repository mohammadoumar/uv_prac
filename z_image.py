import torch
#from diffusers import ZImagePipeline # type: ignore
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig # type: ignore

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs

hf_path = "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/z_image_turbo-Q3_K_M.gguf"

transformer = ZImageTransformer2DModel.from_single_file(
    hf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)


pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    #"unsloth/Z-Image-Turbo-GGUF",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

#prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
#prompt = "A refined, minimalist silhouette logo for a high-end luxury automotive brand, inspired by the elegance of the Mercedes-Benz S-Class. Use sleek lines, subtle brushstrokes, and soft lighting to convey relaxed sophistication, comfort, and prestige. The design should evoke effortless speed, modern automotive living, and the feeling of fulfilled aspirations—quiet luxury rather than excess. Incorporate elements that suggest smooth motion and aerodynamic efficiency, while maintaining a clean and uncluttered aesthetic. The color palette should be understated yet impactful, utilizing shades that reflect timeless elegance and modernity. Overall, the logo should embody the essence of a premium automotive experience, appealing to discerning customers who value both performance and refined design."
#prompt = "A refined logo for a high-end luxury automotive brand, inspired by the elegance of the Mercedes-Benz S-Class. Use sleek lines, subtle brushstrokes, and soft lighting to convey relaxed sophistication, comfort, and prestige. The design should evoke effortless speed, modern automotive living, and the feeling of fulfilled aspirations—quiet luxury rather than excess."
prompt = "Logo for a high-end, luxury cars company, with Mercedes S Series being the main offering. Color, lines, brushstrokes and lights to exude a sense of relaxed luxury and fulfilled dreams. A sense of speed and comfort in modern automobile living."
print(prompt)
# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example_logo_2z.png")
