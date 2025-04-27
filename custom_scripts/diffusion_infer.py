from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import cv2

img_path = "/home/yuchunli/_DATASET/caltech101/101_ObjectCategories/airplanes/image_0012.jpg"

# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", 
#     torch_dtype=torch.float16, 
#     use_safetensors=True, 
#     variant="fp16"
# )
# pipe.to("cuda")


# init_image = cv2.imread(
#     "/home/yuchunli/_DATASET/caltech101/101_ObjectCategories/airplanes/image_0012.jpg"
# )

# init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)

# # strength ∈ [0.0–1.0]: lower = more like the original
# result = pipe(
#     prompt="make this image look like it's pixelated in low resolution",
#     image=init_image,
#     num_inference_steps=1000,
#     # strength=0.1,
#     # guidance_scale=9.0,
# ).images[0]

# result.save("img2img_output.png")


#### Style Transfer ####

# load pipeline
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

# set the adapter and scales - this is a component that lets us add the style control from an image to the text-to-image model
scale = {
    "down": {"block_2": [0.0, 1.0]},
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)

style_image = load_image(img_path)

generator = torch.Generator(device="cpu").manual_seed(26)
image = pipeline(
    prompt="pixelated style, nice pixelated style, pixel art, pixelated, pixelated character, pixelated background, pixelated landscape",
    ip_adapter_image=style_image,
    negative_prompt="low contrast, blurry, deformed, glitch, low quality, worst quality",
    guidance_scale=5,
    num_inference_steps=30,
    generator=generator,
).images[0]

image.save("ip_adapter_output.png")

# pipe = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", 
#     torch_dtype=torch.float16, 
#     use_safetensors=True, 
#     variant="fp16"
# )
# pipe.to("cuda")

# # strength ∈ [0.0–1.0]: lower = more like the original
# result = pipe(
#     prompt="generate a airplane in the sky in pixelated style",
#     num_inference_steps=1000,
#     # strength=0.7,
#     # guidance_scale=7.5
# ).images[0]

# result.save("text2img_output.png")