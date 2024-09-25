from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("lora_weights_1_4_100_1000_0.0001", weight_name="pytorch_lora_weights.safetensors")
image = \
    pipe(prompt="vector art of a Bangladeshi man wearing blue T-shirt feeling cold putting his arms around him").images[
        0]
image.save("symptom_output8.png")
