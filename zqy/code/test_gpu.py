import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU available.")

with open("/puhome/24112456g/kaggleMATH/zqy/code/111", "w") as f:
    f.write("adlafladna")

