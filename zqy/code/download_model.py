import os

import pandas as pd
import polars as pl

from transformers import AutoModelForCausalLM, AutoTokenizer

save_directory = "/puhome/24112456g/kaggleMATH/zqy/temp/qwen72b"
model_name = "Qwen/Qwen2.5-Math-72B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# save model

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)