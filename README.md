## This is an article repository. It is in the process of being updated while the paper is being reviewed.
## Title: Large Language Models for Predicting Organic Synthesis Procedures
Available online: -- 

Required: python 3.11; pytorch 2.2; transformers 4.38.1. GPU with support for CUDA 11.8 and cudnn 8.9.2 is recommended. Large models need 24GB Vram GPU's to train. We used vast.ai and RTX4090.

## Quickstart; try the molT5-large fine-tuned model.
```python
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('laituan245/molt5-large-smiles2caption',model_max_length=256)
model = T5ForConditionalGeneration.from_pretrained('.../molT5/pre-tok_pre_molT5-large/checkpoint-17000')
model.config.max_length = 512
model.to(torch_device)

#The reactants and products are separated by the bar (|)
input = 'Clc1nc(Cl)c2c(n1)CSC2|C1COCCN1>>Clc1nc2c(c(N3CCOCC3)n1)SCC2'   
input_enc = tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(torch_device)

output = model.generate(**input_enc,max_new_tokens=256, num_beams=3, early_stopping=True)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### ----------------------------------------

## Datasets 
The prepared "PRP-931k" dataset can be downloaded [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EZ-t5XCV3qlAoKWDYUUbYRkBNTtInoDSt20XQIuYwCpGtA?e=5Mtbm2). (100 MB compressed, .txt files, contains train, test and validation subsets)

## Best models 

Download "molT5-large-PP" model (best) [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EdxiYDTRyZROoQ70FbsajwkBOqIHiSAhrGM2Uqanhfd40g?e=OTGkSq) (8GB compressed) 

Download "molT5-base-PP" model (faster) [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EVV5glzozlJDn43OuKmCkDgBPGesVw6f6eVm41iBXZPqRg?e=fKL31e) (2.5GB compressed)

Download "FLAN-T5-base-CC" model (faster) [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EcxBFVpmP85OmxX95Rb-z9UBT1NNQITXKP6EFTPhWXphEQ?e=EYPIR6) (2.6GB compressed)

Download "molT5-large-PP" model (faster) [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EWL-0ZVh7vpMq2XrQIWVGuYBiKyGKiRNigOAQRP2XHo6QQ?e=4kZPEe) (2.8GB compressed)

## All other models

Download all mol-T5 models [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EffPwF00BKFAp4M2vRfR5CgBphSxE5giHGygrJx80F63tQ?e=6bJgH8) (11.4GB compressed) 

Download all FLAN-T5 models [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/ETnG5n3B1hZHnjyTjjN3I0ABrCYPMdDS4mJJseyNXfKlWA?e=iOL26r) (18.2GB compressed) 

Download all BART models [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EdsWZC-08rpBn4o8ZAtVvdEBsxxjG8jbdFvfP4c-d-IQog?e=YJwAUK) (9.6GB compressed) 

Download all T5 models [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EVmV9qOsR1hMlIOUV6vv3EgBTUYcqmCBRB0z_5nGvaleIQ?e=sNq7Gz) (9GB compressed) 

Download all seq2seq transformer models [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EbKFzqs3-udBj8pK8fg_9GYB3XwomthKmvP1aGUUyUhZhg?e=ZcT67m) (3.9GB compressed) 

### ----------------------------------------

## Other files

Download the DRFP pkl files for tmap [click here](https://vduedu-my.sharepoint.com/:u:/g/personal/mantas_vaskevicius_vdu_lt/EUX4P8mWIpVMqhibomR0bR8ByE3xKMmAKrmfNTdEfMoEPg?e=DEcCoK)

### ----------------------------------------

The 'paragraph2actions_updated' files contain code for loading up actions and parsing the formal (coverted) format of the paragraphs. We share the modified version of the IBM RXN's original.
