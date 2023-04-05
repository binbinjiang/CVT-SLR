import torch
import random
import numpy as np


class RandomState(object):
    def __init__(self, seed):
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def save_rng_state(self):
        rng_dict = {}
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        rng_dict["numpy"] = np.random.get_state()
        rng_dict["random"] = random.getstate()
        return rng_dict

    def set_rng_state(self, rng_dict):
        torch.set_rng_state(rng_dict["torch"])
        torch.cuda.set_rng_state_all(rng_dict["cuda"])
        np.random.set_state(rng_dict["numpy"])
        random.setstate(rng_dict["random"])
