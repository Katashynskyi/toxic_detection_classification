import os

os.makedirs("data", exist_ok=True)
for model_name in ["concatenate", "image", "text", "glove"]:
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(f"result/{model_name}", exist_ok=True)
