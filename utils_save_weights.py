
from customing_marian import MarianMTModel
from tokenizing_marian import MarianTokenizer
import torch

model_dir = "opus-mt-fr-en"

model = MarianMTModel.from_pretrained(model_dir)
tokenizer = MarianTokenizer.from_pretrained(model_dir)

torch.save(model.state_dict(), f"{model_dir}/weights.pt")
