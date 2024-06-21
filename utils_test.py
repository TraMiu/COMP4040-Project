
from customing_marian import MarianMTModel
from tokenizing_marian import MarianTokenizer
import torch
from transformers import MarianConfig

model_dir = "opus-mt-vi-en"

tokenizer = MarianTokenizer.from_pretrained(model_dir)

config = MarianConfig.from_pretrained(model_dir)
base_model = MarianMTModel(config)
loaded_weights = torch.load(f"{model_dir}/weights.pt")

weight_model = MarianMTModel(config)
weight_model.load_state_dict(loaded_weights, strict=False)

strict_weight_model = MarianMTModel(config)
strict_weight_model.load_state_dict(loaded_weights, strict=True)

model = MarianMTModel.from_pretrained(model_dir)


sample_text = "où est l'arrêt de bus ?"
sample_text = "Bến xe buýt ở đâu vậy?"
bahnar_text = "Ih đei hlôi 'bĭch kơpal gyương 'nâu đunh dêh bơih."
sample_text = "Bạn đã nằm trên chiếc giường này rất lâu rồi."
batch = tokenizer([sample_text], return_tensors="pt")

print("Original models:")
generated_ids = weight_model.generate(**batch)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

generated_ids = model.generate(**batch)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

generated_ids = strict_weight_model.generate(**batch)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()



print("Customed models:")
generated_ids = weight_model.generate(**batch, is_tgt=True)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

generated_ids = model.generate(**batch, is_tgt=True)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

generated_ids = strict_weight_model.generate(**batch, is_tgt=True)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()

