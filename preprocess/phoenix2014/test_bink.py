import numpy as np

gloss_dict = np.load("/root/slt_proj/VAC_CSLR/preprocess/phoenix2014T/gloss_dict.npy", allow_pickle=True).item()
num_classes = len(gloss_dict) + 1

print(num_classes)