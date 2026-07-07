import pronouncing 
from common import *
text = "autumn"
pronouncing.phones_for_word("it's all")

ckpt_path =  "./data/finetune-model.pt"
user_dir = "./library"

dictionary = extract_dictionary(ckpt_path, user_dir)
tgt_list = dictionary.symbols

for token in tgt_list:
    if len(token) <=2:
        print(token)

with open("results/avh_dictionary.pkl", "wb") as f:
            pickle.dump(tgt_list, f)
