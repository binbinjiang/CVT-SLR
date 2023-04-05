import os
import yaml
import torch
import importlib
import faulthandler
import numpy as np
from collections import OrderedDict

faulthandler.enable()
import utils
from seq_scripts import seq_eval

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model = self.loading()

    def eval(self):
        if self.arg.load_weights is None and self.arg.load_checkpoints is None:
            raise ValueError('Please appoint --load-weights.')
        self.recoder.print_log('Model:   {}.'.format(self.arg.model))
        self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
        
        # eval dev
        dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                            "dev", 6667, self.arg.work_dir, self.recoder)
        
        # eval test
        test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                            "test", 6667, self.arg.work_dir, self.recoder)
        
        self.recoder.print_log('Evaluation Done.\n')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            args=self.arg
        )

        self.load_model_weights(model, self.arg.load_weights)
        model = model.cuda()
        
        self.load_data()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        weights = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict['model_state_dict'].items()])

        if self.arg.use_seqAE.lower()=="vae":
            model.load_state_dict(weights, strict=True)
        else:
            model.load_state_dict(weights, strict=False)


    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        # print(key, default_arg.keys())
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)

    # for evaluation
    processor.eval()
