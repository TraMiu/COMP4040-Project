#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### Wandb session

# In[ ]:


import wandb
wandb.login(key='')
wandb.init(project='huggingface')


# # Customing marian
# Description of the model format a.b.c
# * **a** means the phase of the dataset. Phase 1 means the first 1000 high-quality English-Bahnar sentences
# * **b** means the direction of translation. Direction 1 means Bahnar to English
# * **c** means the special technique and version to develop this model.
# 
# Technique 0 means just feed English-Bahnar data into the pretrained model. Technique 1 means feed both English-Bahnar and English-Vietnamese into the model
# 

# ### Installing dependencies

# In[ ]:


# !pip install -q -U sentencepiece
# !pip install -q -U datasets
# !pip install -q -U sacrebleu


# ### Importing libraries

# In[ ]:


import torch
import sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from datasets import load_dataset, load_metric
from customing_marian import MarianMTModel
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, concatenate_datasets

import sacrebleu


# In[ ]:


from datasets import Dataset, concatenate_datasets


# ### Reproducibility

# In[ ]:


RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
# If you are using CUDA
torch.cuda.manual_seed_all(RANDOM_SEED)
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PADDING_LEN = 228
IS_CUSTOMED = False


# ### Loading the pretrained model

# In[ ]:


model_name = "opus-mt-en-vi"  # Example model for Bahnar to English
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)


# ### Loading the dataset

# In[ ]:


with open('data/KJV-Bible-ba-en/bdq-eng.train.bdq', 'r', encoding='utf-8') as file_ba:
    ba_data = file_ba.readlines()
with open('data/KJV-Bible-ba-en/bdq-eng.train.eng', 'r', encoding='utf-8') as file_en:
    en_data = file_en.readlines()
assert len(ba_data) == len(en_data), "The files don't have the same number of lines."
train_df = pd.DataFrame({'English': en_data, 'Bahnar': ba_data})
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED)

ba_en_train_dataset = Dataset.from_pandas(train_df)
ba_en_val_dataset = Dataset.from_pandas(val_df)

with open('data/PhoMT-vi-en/dev.vi', 'r', encoding='utf-8') as file_vi:
    vi_data = file_vi.readlines()
with open('data/PhoMT-vi-en/dev.en', 'r', encoding='utf-8') as file_en:
    en_data = file_en.readlines()
assert len(vi_data) == len(en_data), "The files don't have the same number of lines."
train_df = pd.DataFrame({'English': en_data, 'Vietnamese': vi_data})

vi_en_train_dataset = Dataset.from_pandas(train_df)


# In[ ]:


with open('data/Doccano-ba-en/bdq-eng.test.bdq', 'r', encoding='utf-8') as file_ba:
    ba_data = file_ba.readlines()
with open('data/Doccano-ba-en/bdq-eng.test.eng', 'r', encoding='utf-8') as file_en:
    en_data = file_en.readlines()
assert len(ba_data) == len(en_data), "The files don't have the same number of lines."
test_df = pd.DataFrame({'English': en_data, 'Bahnar': ba_data})
ba_en_test_dataset = Dataset.from_pandas(test_df)

print(len(ba_en_train_dataset), len(ba_en_val_dataset), len(ba_en_test_dataset))
print(len(vi_en_train_dataset))


# # Preprocess the dataset with tokenizer

# In[ ]:


def preprocess_function_source(examples):
    inputs = examples['English']
    targets = examples['Vietnamese']
    model_inputs = tokenizer(inputs, max_length=PADDING_LEN, padding='max_length', truncation=True)

    # Set up the tokenizer for Vietnamese
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=PADDING_LEN, padding='max_length', truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_vi_en = vi_en_train_dataset.map(preprocess_function_source, batched=True)


# In[ ]:


for example in tokenized_vi_en:
    example['is_src'] = IS_CUSTOMED


# In[ ]:


def preprocess_function_target(examples):
    inputs = examples['English']
    targets = examples['Bahnar']
    model_inputs = tokenizer(inputs, max_length=PADDING_LEN, padding='max_length', truncation=True)

    # Set up the tokenizer for Vietnamese
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=PADDING_LEN, padding='max_length', truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ba_en = ba_en_train_dataset.map(preprocess_function_target, batched=True)
tokenized_ba_en_val = ba_en_val_dataset.map(preprocess_function_target, batched=True)


# In[ ]:


for example in tokenized_ba_en:
    example['is_tgt'] = IS_CUSTOMED
for example in tokenized_ba_en_val:
    example['is_tgt'] = IS_CUSTOMED


# ### Concatenate Vietnamese-English and Bahnar-English Dataset

# In[ ]:


tokenized_viAndba_en = concatenate_datasets([tokenized_vi_en, tokenized_ba_en ])
# viOrba_en_dataset is now a combined dataset
print(tokenized_viAndba_en)


# In[ ]:


#shuffled_dataset = tokenized_viAndba_en.shuffle(seed=RANDOM_SEED)


# 
# # Train the model

# In[ ]:


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract the `is_tgt` flag from inputs or set it to False if not present
        is_tgt = inputs.pop('is_tgt', False)
        is_src = inputs.pop('is_src', False)

        outputs = model(**inputs, is_tgt=is_tgt, is_src=is_src)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    

print((len(tokenized_viAndba_en)))
# Use data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Step 3: Fine-Tuning the Model on Bahnar to English
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.015,  # L2 regularization
    save_total_limit=3,
)


trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_viAndba_en,
    eval_dataset=tokenized_ba_en_val,
    data_collator=data_collator,
)

trainer.train()


# # Test the model

# In[ ]:


import torch
from tqdm.auto import tqdm

def evaluate_model(model, encode_tokenizer, decode_tokenizer, dataset, device):
    model.eval()
    prt = True

    predictions, references = [], []
    for example in tqdm(dataset, desc="Translating"):
        with encode_tokenizer.as_target_tokenizer():
            input_ids = encode_tokenizer.encode(example['English'], return_tensors='pt').to(device)

        output_ids = model.generate(input_ids, is_tgt=IS_CUSTOMED)[0]
        pred = decode_tokenizer.decode(output_ids, skip_special_tokens=True)
        predictions.append([pred])

        references.append([example['Bahnar']])

    return predictions, references


# In[ ]:


test_examples = [{'Bahnar': ex['Bahnar'], 'English': ex['English']} for ex in ba_en_test_dataset]
predictions, references = evaluate_model(model, tokenizer, tokenizer, test_examples, device)


# In[ ]:


predictions[0], references[0] # Check format of predictions and references


# In[ ]:


len(predictions), len(references)


# ### Calculate sacreBLEU

# In[ ]:


preds = []
refs = []
for pred in predictions:
    preds.append(pred[0])
    
for ref in references:
    refs.append(ref)

print(refs[:2])
print(preds[:2])


# In[ ]:


# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds, refs)
print("BLEU: ", round(bleu.score, 2))

# Calculate CHRF
chrf = sacrebleu.corpus_chrf(preds, refs)
print("CHRF:", round(chrf.score, 2))

# Calculate TER
metric = sacrebleu.metrics.TER()
ter = metric.corpus_score(preds, refs)
print("TER:", round(ter.score, 2))


# ### Saving the model

# In[ ]:


# Step 5: Save the Model
model.save_pretrained("./model-en-ba")
tokenizer.save_pretrained("./tokenizer-en-ba")


# ### Display translated and references for qualitative evaluation

# In[ ]:


# Display the first 3 prediction-reference pairs
for i in range(20):
    print(f"Prediction {i+1}: {predictions[i][0]}")
    print(f"Reference {i+1}: {references[i][0]}")

