import argparse
import torch
from diffusers import DDPMPipeline
from PIL import Image
import os

# Load the fine-tuned model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with a fine-tuned DDPM model")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save generated images")
    parser.add_argument("--ckpt_path", type=str, default="finetuned-ddpm-cifar10", help="Path to the fine-tuned model checkpoint")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the fine-tuned model
    pipeline = DDPMPipeline.from_pretrained(
        args.ckpt_path,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate images
    generator = torch.Generator(device=pipeline.device).manual_seed(42)
    output = pipeline(
        batch_size=args.num_images,
        generator=generator,
        num_inference_steps=1000,
        output_type="pil"
    )

    # Save the images
    for i, img in enumerate(output.images):
        img.save(f"{args.output_dir}/sample_{i}.png")

    print(f"Saved {len(output.images)} samples to {args.output_dir}/")