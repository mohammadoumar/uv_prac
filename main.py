import torch
import argparse

from scripts.generate import generate
from transformers import AutoModelForCausalLM

argparser = argparse.ArgumentParser()
argparser.add_argument('--image', type=str, required=True, help='The input image')
argparser.add_argument('--task', type=str, required=True, choices=['caption', 'query', 'reasonquery'], help='Query to ask the model about the image')
args = argparser.parse_args()

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda", # "cuda" on Nvidia GPUs
)

print(generate(model, args.image, args.task))