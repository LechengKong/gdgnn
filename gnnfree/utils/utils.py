from itertools import product
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import random
import os
import json


class SmartTimer:
    def __init__(self, verb=True) -> None:
        self.last = time.time()
        self.verb = verb

    def record(self):
        self.last = time.time()

    def cal_and_update(self, name):
        now = time.time()
        if self.verb:
            print(name, now - self.last)
        self.record()


def get_rank(b_score):
    order = np.argsort(b_score)
    return len(order) - np.where(order == 0)[0][0]


def save_params(filename, params):
    d = vars(params)
    with open(filename, "a") as f:
        json.dump(d, f)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sample_variable_length_data(rown, coln, row_ind, col_ind):
    # print(row_ind)
    keep_arr = np.zeros((rown, coln + 2))
    row_size_count = np.bincount(row_ind, minlength=rown)
    row_cum = np.cumsum(row_size_count)
    fill_size_count = np.clip(row_size_count, 0, coln)
    keep_arr[:, 0] = 1
    keep_arr[np.arange(len(row_size_count)), fill_size_count + 1] = -1
    keep_arr = np.cumsum(keep_arr, axis=-1)[:, 1:-1]
    keep_row, keep_col = keep_arr.nonzero()
    shuffle_ind = row_size_count > coln
    ind_arr = np.zeros_like(keep_arr, dtype=int)
    ind_arr[:] = np.arange(coln)
    if shuffle_ind.sum() > 0:
        select_ind = np.random.choice(
            1500, size=(np.sum(shuffle_ind), coln), replace=False
        )
        select_ind = select_ind % row_size_count[shuffle_ind][:, None]
        ind_arr[shuffle_ind] = select_ind
    if rown > 1:
        ind_arr[1:] += row_cum[:-1, None]
    sampled_ind = ind_arr[keep_row, keep_col]
    sample_val = col_ind[sampled_ind]
    sampled_arr = np.zeros_like(keep_arr)
    sampled_arr[keep_row, keep_col] = sample_val
    return sampled_arr, keep_arr


def k_fold_ind(labels, fold):
    ksfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=10)
    folds = []
    for _, t_index in ksfold.split(
        np.zeros_like(np.array(labels)), np.array(labels, dtype=int)
    ):
        folds.append(t_index)
    return folds


def k_fold2_split(folds, data_len):
    splits = []
    for i in range(len(folds)):
        test_arr = np.zeros(data_len, dtype=bool)
        test_arr[folds[i]] = 1
        val_arr = np.zeros(data_len, dtype=bool)
        val_arr[folds[int((i + 1) % len(folds))]] = 1
        train_arr = np.logical_not(np.logical_or(test_arr, val_arr))
        train_ind = train_arr.nonzero()[0]
        test_ind = test_arr.nonzero()[0]
        val_ind = val_arr.nonzero()[0]
        splits.append([train_ind, test_ind, val_ind])
    return splits


def dict_res_summary(res_col):
    res_dict = {}
    for res in res_col:
        for k in res:
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(res[k])
    return res_dict


def multi_data_average_exp(data, args, exp):
    val_res_col = []
    test_res_col = []
    for split in data:
        val_res, test_res = exp(split, args)
        val_res_col.append(val_res)
        test_res_col.append(test_res)

    val_res_dict = dict_res_summary(val_res_col)
    test_res_dict = dict_res_summary(test_res_col)
    return val_res_dict, test_res_dict


def hyperparameter_grid_search(
    hparams, data, exp, args, search_metric, evaluator, exp_arg=None
):
    named_params = [[(k, p) for p in hparams[k]] for k in hparams]
    best_met = evaluator.init_result()
    best_res = None
    params = product(*named_params)
    for p in params:
        for name, val in p:
            setattr(args, name, val)
        if exp_arg:
            val_res, test_res = exp(data, args, exp_arg)
        else:
            val_res, test_res = exp(data, args)
        val_metric_res, test_metric_res = np.array(
            val_res[search_metric]
        ), np.array(test_res[search_metric])
        val_mean, val_std = np.mean(val_metric_res), np.std(val_metric_res)
        test_mean, test_std = np.mean(test_metric_res), np.std(test_metric_res)
        if evaluator.better_results(val_mean, best_met):
            best_met = val_mean
            best_res = {
                "metric": search_metric,
                "val_mean": val_mean,
                "val_std": val_std,
                "test_mean": test_mean,
                "test_std": test_std,
                "full_val": val_res,
                "full_test": test_res,
                "params": p,
            }
    return best_res


def write_res_to_file(
    file,
    dataset,
    metric,
    test_mean,
    val_mean=0,
    test_std=0,
    val_std=0,
    params=None,
    res=None,
):
    with open(file, "a") as f:
        f.write("\n\n")
        f.write(res)
        f.write("\n")
        f.write("Dataset: {} \n".format(dataset))
        f.write("Optimize wrt {}\n".format(metric))
        f.write("val, {:.5f} ± {:.5f} \n".format(val_mean, val_std))
        f.write("test, {:.5f} ± {:.5f} \n".format(test_mean, test_std))
        f.write("best res:")
        f.write(str(params))


def var_size_repeat(size, chunks, repeats):
    a = np.arange(size)
    s = np.r_[0, chunks.cumsum()]
    starts = a[np.repeat(s[:-1], repeats)]
    if len(starts) == 0:
        return np.array([], dtype=int)
    chunk_rep = np.repeat(chunks, repeats)
    ends = starts + chunk_rep

    clens = chunk_rep.cumsum()
    ids = np.ones(clens[-1], dtype=int)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
    out = ids.cumsum()
    return out


def count_to_group_index(count):
    return torch.arange(len(count), device=count.device).repeat_interleave(
        count
    )
