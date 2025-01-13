import pickle
from drfp import DrfpEncoder

with open("full_v1_combined.txt", "r",encoding='utf-8') as f:
    input_data_train = [line.strip().replace('\u200c','').split('\t')[0] for line in f.readlines()]

fingerprints = DrfpEncoder.encode(input_data_train,show_progress_bar=True)

with open('full_v1_combined_FP.pkl', "wb+") as f:
    pickle.dump(fingerprints, f)