# This is an article repository. It is in the process of being updated while the paper is being reviewed.
# Title: Large Language Models for Predicting Organic Synthesis Procedures
Available online: -- 

Required: python 3.11; pytorch 2.2; transformers 4.38.1. GPU with support for CUDA 11.8 and cudnn 8.9.2 is recommended. Large models need 24GB Vram GPU's to train. We used vast.ai and RTX4090.

# Quickstart to try the molT5-large fine-tuned model.
```python
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('laituan245/molt5-large-smiles2caption',model_max_length=256)
model = T5ForConditionalGeneration.from_pretrained('.../molT5/pre-tok_pre_molT5-large/checkpoint-17000')
model.config.max_length = 512

#The reactants and products are separated by the bar (|)
input = 'Clc1nc(Cl)c2c(n1)CSC2|C1COCCN1>>Clc1nc2c(c(N3CCOCC3)n1)SCC2'   
input_enc = tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(torch_device)

output = model.generate(**input_enc,max_new_tokens=256, num_beams=3, early_stopping=True)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### ----------------------------------------
## Datasets and models 

