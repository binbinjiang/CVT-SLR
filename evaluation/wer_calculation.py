import numpy as np
from itertools import groupby
import os

def load_groundtruth(fpath):
    file_info = open(fpath, 'r', encoding='utf-8').readlines()
    gt_dict = dict()
    for line in file_info:
        info = line[:-1].split(" ")[5:]
        info = [*filter(lambda x: len(x), info)]
        gt_dict[line.split(" ")[0]] = info
    return gt_dict


def load_prediction(fpath):
    file_info = open(fpath, 'r', encoding='utf-8').readlines()
    pre_dict = dict()
    for line in file_info:
        file_name, _, _, _, wd = line[:-1].split(" ")
        if file_name not in pre_dict.keys():
            pre_dict[file_name] = [wd]
        else:
            pre_dict[file_name].append(wd)
    return pre_dict


def get_wer_delsubins(ref, hyp, merge_same=False, align_results=False,
                      penalty={'ins': 1, 'del': 1, 'sub': 1}):
    # whether merge glosses before evaluation
    hyp = hyp if not merge_same else [x[0] for x in groupby(hyp)]

    # initialization
    ref_lgt = len(ref) + 1
    hyp_lgt = len(hyp) + 1

    costs = np.ones((ref_lgt, hyp_lgt), dtype=np.int) * 1e6
    # auxiliary values
    costs[0, :] = np.arange(hyp_lgt) * penalty['ins']
    costs[:, 0] = np.arange(ref_lgt) * penalty['del']

    backtrace = np.zeros((ref_lgt, hyp_lgt), dtype=np.int)
    # auxiliary indexes, 0, 1, 2, 3 are corresponding to correct, substitute, insert and delete, respectively
    backtrace[0, :] = 2
    backtrace[:, 0] = 3

    # dynamic programming
    for i in range(1, ref_lgt):
        for j in range(1, hyp_lgt):
            if ref[i - 1] == hyp[j - 1]:
                costs[i, j] = min(costs[i - 1, j - 1], costs[i, j])
                backtrace[i, j] = 0
            else:
                sub_cost, ins_cost, del_cost = \
                    costs[i - 1, j - 1] + penalty['sub'], \
                    costs[i - 1, j] + penalty['del'], \
                    costs[i, j - 1] + penalty['ins']
                min_cost = min(del_cost, ins_cost, sub_cost)
                if min_cost < costs[i, j]:
                    costs[i, j] = min_cost
                    backtrace[i, j] = [sub_cost, ins_cost, del_cost].index(costs[i, j]) + 1

    # backtrace pointer
    bt_ptr = np.array([ref_lgt - 1, hyp_lgt - 1])
    bt_path = []
    while bt_ptr.min() > 0:
        if backtrace[bt_ptr[0], bt_ptr[1]] == 0:
            # if correct, move (-1, -1)
            bt_ptr = bt_ptr - 1
            op = 'C'
        elif backtrace[bt_ptr[0], bt_ptr[1]] == 1:
            # if substitute, move (-1, -1)
            bt_ptr = bt_ptr - 1
            op = 'S'
        elif backtrace[bt_ptr[0], bt_ptr[1]] == 2:
            # if delete, move (-1, 0)
            bt_ptr = bt_ptr + (-1, 0)
            op = 'D'
        elif backtrace[bt_ptr[0], bt_ptr[1]] == 3:
            # if insert, move (0, -1)
            bt_ptr = bt_ptr + (0, -1)
            op = 'I'
        else:
            assert "Unexpected Operation"
        bt_path.append((bt_ptr, op))

    # decode path
    aligned_gt = []
    aligned_pred = []
    results = []
    for i in range(bt_path[-1][0][0]):
        aligned_gt.append(ref[i])
        aligned_pred.append('*' * len(ref[i]))
        results.append('D' + ' ' * (len(ref[i]) - 1))
    for i in range(bt_path[-1][0][1]):
        aligned_pred.append(hyp[i])
        aligned_gt.append('*' * len(hyp[i]))
        results.append('I' + ' ' * (len(hyp[i]) - 1))
    for ptr, op in bt_path[::-1]:
        if op in ['C', 'S']:
            if align_results:
                delta_lgt = len(ref[ptr[0]]) - len(hyp[ptr[1]])
                ref_pad = 0 if delta_lgt > 0 else -delta_lgt
                hyp_pad = 0 if delta_lgt < 0 else delta_lgt
                aligned_gt.append(ref[ptr[0]] + ' ' * ref_pad)
                aligned_pred.append(hyp[ptr[1]] + ' ' * hyp_pad)
            else:
                aligned_gt.append(ref[ptr[0]])
                aligned_pred.append(hyp[ptr[1]])
        elif op == 'I':
            aligned_gt.append('*' * len(hyp[ptr[1]]))
            aligned_pred.append(hyp[ptr[1]])
        elif op == 'D':
            aligned_gt.append(ref[ptr[0]])
            aligned_pred.append('*' * len(ref[ptr[0]]))

        if op == 'C':
            results.append(' ' * (len(aligned_gt[-1])))
        else:
            results.append(op + ' ' * (len(aligned_gt[-1]) - 1))
    return aligned_gt, aligned_pred


def calculate_stats(gt, lstm_pred):
    stat_ret = {
        'wer_lstm': 0,
        'cnt': 0,
    }
    for i in range(len(gt)):
        if "*" not in gt[i]:
            stat_ret['cnt'] += 1
        if gt[i] != lstm_pred[i]:
            stat_ret['wer_lstm'] += 1
    return stat_ret


def sent_evaluation(**kwargs):
    gt, lstm_pred = get_wer_delsubins(kwargs['gt'], kwargs['lstm_prediction'],
                                      merge_same=kwargs['merge_same'],
                                      penalty=kwargs['penalty'])
    return calculate_stats(gt, lstm_pred)


def sum_dict(dict_list):
    ret_dict = dict()
    for key in dict_list[0].keys():
        ret_dict[key] = sum([d[key] for d in dict_list])
    return ret_dict


def wer_calculation(gt_path, primary_pred):
    gt = load_groundtruth(gt_path)
    pred1 = load_prediction(primary_pred)
    results_list = []
    for fileid, sent in gt.items():
        sent_stat = sent_evaluation(
            info=fileid, gt=sent,
            merge_same=True,
            lstm_prediction=pred1[fileid],
            penalty={'ins': 3, 'del': 3, 'sub': 4},
        )
        results_list.append(sent_stat)
    results = sum_dict(results_list)
    return results['wer_lstm'] / results['cnt'] * 100


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,output_file=None):
    
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")

    ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", f"{prefix}out.{output_file}")
    
    return ret
