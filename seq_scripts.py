import torch
from evaluation.wer_calculation import evaluate
from tqdm import tqdm

def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    model.eval()
    total_sent = []
    total_info = [] 
    for data in tqdm(loader):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1]) 
        with torch.no_grad(): 
            ret_dict = model.eval_network(vid, vid_lgt)

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        
    write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
    ret = evaluate(
        prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
        evaluate_dir=cfg.dataset_info['evaluation_dir'],
        evaluate_prefix=cfg.dataset_info['evaluation_prefix']
    )

    recoder.print_log(f"Epoch {epoch}, {mode} {ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return ret


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines("{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx], word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100, word[0]))
