
from customing_marian_p2 import MarianMTModel
from tokenizing_marian import MarianTokenizer
import torch
from transformers import MarianConfig

model_dir = "opus-mt-vi-en"

config = MarianConfig.from_pretrained(model_dir)
weight_model = MarianMTModel(config)
loaded_weights = torch.load(f"{model_dir}/weights.pt")

tokenizer = MarianTokenizer.from_pretrained(model_dir)
weight_model.load_state_dict(loaded_weights, strict=False)


bahnar_text = "Ih đei hlôi 'bĭch kơpal gyương 'nâu đunh dêh bơih."
vie_text = "Bạn đã nằm trên chiếc giường này rất lâu rồi."


print("Original models:")
print("Total vocab in tokenizer:", len(tokenizer))
for idx, (token, token_id) in enumerate(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)):
    print(f"Special token {idx}: '{token}' with ID: {token_id}")

batch = tokenizer([vie_text], return_tensors="pt")
bahnar_batch = tokenizer([bahnar_text], return_tensors="pt")

weight_model.eval()
generated_ids = weight_model.generate(**batch)
print("Translation from Vietnamese with original vi-en model:\n", tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0], "\n")
# generated_ids = weight_model.generate(**bahnar_batch)
# print("Translation from Bahnar with augmented model:\n", tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0], "\n")
generated_ids = weight_model.generate(**bahnar_batch, is_tgt=True)
print("Translation from Bahnar with augmented model:\n", tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print()



#code to add special token
# num_added_toks = tokenizer.add_tokens(['<aug>'], special_tokens=True) ##This line is updated
# weight_model.resize_token_embeddings(len(tokenizer))

# batch = tokenizer([vie_text], return_tensors="pt")
# bahnar_batch = tokenizer([bahnar_text], return_tensors="pt")

# print("Added special token models:")
# print("Total vocab in tokenizer:", len(tokenizer))
# for idx, (token, token_id) in enumerate(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)):
#     print(f"Special token {idx}: '{token}' with ID: {token_id}")

# generated_ids = weight_model.generate(**batch)
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
# generated_ids = weight_model.generate(**bahnar_batch, is_tgt=True)
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
# print()

# weight_model.save_pretrained(f"{model_dir}+Stoken")
# tokenizer.save_pretrained(f"{model_dir}+Stoken")
