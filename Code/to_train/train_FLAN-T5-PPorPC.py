import random
from torch.utils.data import Dataset, DataLoader
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, AdamW, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
import torch
import numpy as np
import evaluate
from nltk.translate.bleu_score import corpus_bleu

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Define dataset classes
class Train_Dataset(Dataset):
    def __init__(self, input_data, output_data, tokenizer, max_length=256):
        self.input_data = input_data
        self.output_data = output_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        inp_list = self.input_data[idx]
        out_data = self.output_data[idx]
        input_encodings = self.tokenizer(inp_list, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        output_encodings = self.tokenizer(out_data, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        return (input_encodings["input_ids"][0], input_encodings["attention_mask"][0], output_encodings["input_ids"][0])

    def __len__(self):
        return len(self.input_data)

class Val_Dataset(Dataset):
    def __init__(self, input_data, output_data, tokenizer, max_length=256):
        self.input_data = input_data
        self.output_data = output_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        input_encodings = self.tokenizer(self.input_data[idx], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        output_encodings = self.tokenizer(self.output_data[idx], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        return (input_encodings["input_ids"][0], input_encodings["attention_mask"][0], output_encodings["input_ids"][0])

    def __len__(self):
        return len(self.input_data)

# Load and preprocess data
with open("train_data/X_w2_nonaug_train_f_aug4.txt", "r", encoding='utf-8') as f:
    train_input_data = [line.strip() for line in f.readlines()]
with open("train_data/Y_w2_nonaug_train_f_aug4.txt", "r", encoding='utf-8') as f:
    train_output_data = [line.strip().replace('\u200c', '') for line in f.readlines()]
with open("train_data/X_w2_nonaug_valid_2_f.txt", "r", encoding='utf-8') as f:
    val_input_data = [line.strip() for line in f.readlines()]
with open("train_data/Y_w2_nonaug_valid_2_f.txt", "r", encoding='utf-8') as f:
    val_output_data = [line.strip().replace('\u200c', '') for line in f.readlines()]

# Shuffle training data
temp = list(zip(train_input_data, train_output_data))
random.shuffle(temp)
train_input_data, train_output_data = zip(*temp)
train_input_data, train_output_data = list(train_input_data), list(train_output_data)

# Load pre-trained model and tokenizer


### FOR PRETRAINED MODEL INIT (PP)

# Change to 'google/flan-t5-small' or 'google/flan-t5-large' for other size models
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
model.config.max_length = 256

### FOR CUSTOM MODEL INIT (PC)
# Change to 'google/flan-t5-small' or 'google/flan-t5-base' for other size models
#config = AutoConfig.from_pretrained('google/flan-t5-large')
#model = T5ForConditionalGeneration(config)
#model.config.max_length = 256



tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

# Set up logging and evaluation metrics
steps_done = 0
metric = evaluate.load("sacrebleu")

# Define evaluation functions
def modified_bleu(truth: List[str], pred: List[str]) -> float:
    references = [sentence.split() for sentence in truth]
    candidates = [sentence.split() for sentence in pred]
    references = [r + max(0, 4 - len(r)) * [''] for r in references]
    candidates = [c + max(0, 4 - len(c)) * [''] for c in candidates]
    refs = [[r] for r in references]
    return corpus_bleu(refs, candidates)

def postprocess_text(preds, labels):
    preds = [list(pred.split('</s>'))[0].replace('<unk>', '') for pred in preds]
    labels = [list(label.split('</s>'))[0].replace('<unk>', '') for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    global steps_done
    steps_done += 500
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_preds_save = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds2, decoded_labels2 = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    result["bleu_modified"] = modified_bleu(decoded_labels, decoded_preds) * 100
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Create datasets and data loaders
train_dataset = Train_Dataset(train_input_data, train_output_data, tokenizer)
val_dataset = Val_Dataset(val_input_data, val_output_data, tokenizer)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='main_multi_run/'+name_of_file,
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    learning_rate=0.0001,
    evaluation_strategy='steps',
    predict_with_generate=True,
    eval_steps=500,
    fp16=False,
    # set to 0 if you run into problems
    dataloader_num_workers=2,
    logging_steps=25,
    warmup_steps=0,
    save_total_limit=2,
    gradient_accumulation_steps=32,
    eval_accumulation_steps=1,
)

# Define the trainer object
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=lambda data: {'input_ids': torch.stack([item[0] for item in data]),
                                'attention_mask': torch.stack([item[1] for item in data]),
                                'labels': torch.stack([torch.where(item[2] != tokenizer.pad_token_id, item[2], -100) for item in data])},
)

# Start training
trainer.train()
