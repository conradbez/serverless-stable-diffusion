# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import pipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pipeline('fill-mask', model='bert-base-uncased')
    
    # from diffusers import StableDiffusionPipeline
    # import torch
    # model_id = "prompthero/openjourney"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    # prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"
    # image = pipe(prompt).images[0]
    # image.save("./retro_cars.png")

if __name__ == "__main__":
    download_model()
    