from enum import verify

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
        "./models/stable-diffusion-v1-4",
        local_files_only=True
)
pipe = pipe.to("cpu")
image = pipe("futuristic city, neon lights, rainy night, Blade Runner style").images[0]
image.save("output.png")