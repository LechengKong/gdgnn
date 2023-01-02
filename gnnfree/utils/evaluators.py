import numpy as np
import torch
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from gnnfree.utils import get_rank
from sklearn import metrics as met


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def collect_res(self, res):
        pass

    @abstractmethod
    def summarize_res(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def better_results(self, a, b):
        pass

    @abstractmethod
    def init_result(self):
        pass


class MaxEvaluator(Evaluator):
    def better_results(self, a, b):
        return a > b

    def init_result(self):
        return -1e10


class MinEvaluator(Evaluator):
    def better_results(self, a, b):
        return a < b

    def init_result(self):
        return 1e10


class HeadPosRankingEvaluator(MaxEvaluator):
    def __init__(self, name, neg_sample_size) -> None:
        super().__init__(name)
        self.rankings = []
        self.neg_sample_size = neg_sample_size

    def collect_res(self, res):
        score_mat = res.view(-1, self.neg_sample_size + 1).cpu().numpy()
        score_sort = np.argsort(score_mat, axis=1)
        rankings = len(score_mat[0]) - np.where(score_sort == 0)[1]
        self.rankings += rankings.tolist()

    def summarize_res(self):
        metrics = {}
        rankings = np.array(self.rankings)
        metrics["h10"] = np.mean(rankings <= 10)
        metrics["h3"] = np.mean(rankings <= 3)
        metrics["h1"] = np.mean(rankings == 1)
        metrics["mrr"] = np.mean(1 / rankings)
        return metrics

    def reset(self):
        self.rankings = []


class VarSizeRankingEvaluator(MaxEvaluator):
    def __init__(self, name, num_workers=20) -> None:
        super().__init__(name)
        self.score_list = []
        self.num_workers = num_workers

    def collect_res(self, res, sample_len):
        sample_len = sample_len.cpu().numpy()
        score = res.cpu().numpy().flatten()
        cur_head_pointer = 0
        for sl in sample_len:
            next_hp = cur_head_pointer + sl
            s_score = score[cur_head_pointer:next_hp]
            cur_head_pointer = next_hp
            self.score_list.append(s_score)

    def summarize_res(self):
        metrics = {}
        with mp.Pool(processes=self.num_workers) as p:
            all_rankings = p.map(get_rank, self.score_list)
        rankings = np.array(all_rankings)
        metrics["h1000"] = np.mean(rankings <= 1000)
        metrics["h100"] = np.mean(rankings <= 100)
        metrics["h10"] = np.mean(rankings <= 10)
        metrics["h3"] = np.mean(rankings <= 3)
        metrics["h1"] = np.mean(rankings == 1)
        metrics["mrr"] = np.mean(1 / rankings)
        return metrics

    def reset(self):
        self.score_list = []


class BinaryHNEvaluator(MaxEvaluator):
    def __init__(self, name, hn=None) -> None:
        super().__init__(name)
        self.targets = []
        self.scores = []
        self.hn = hn

    def collect_res(self, res, labels):
        self.scores.append(res.cpu().numpy())
        self.targets.append(labels.cpu().numpy())

    def summarize_res(self):
        metrics = {}
        all_scores = np.concatenate(self.scores)
        all_targets = np.concatenate(self.targets)
        if len(all_targets.shape) <= 1 or len(all_scores.shape) <= 1:
            all_scores = all_scores.reshape(len(all_scores), 1)
            all_targets = all_targets.reshape(len(all_targets), 1)
        ap_list = []
        auc_list = []
        is_labeled = np.logical_not(np.isnan(all_targets))
        for i in range(all_scores.shape[1]):
            if (
                np.sum(all_targets[:, i] == 1) > 0
                and np.sum(all_targets[:, i] == 0) > 0
            ):
                ap_list.append(
                    met.average_precision_score(
                        all_targets[is_labeled[:, i], i],
                        all_scores[is_labeled[:, i], i],
                    )
                )
                auc_list.append(
                    met.roc_auc_score(
                        all_targets[is_labeled[:, i], i],
                        all_scores[is_labeled[:, i], i],
                    )
                )
        metrics["auc"] = sum(auc_list) / len(auc_list)
        metrics["apr"] = sum(ap_list) / len(ap_list)
        if self.hn is not None:
            sort_ind = np.argsort(all_scores)
            ranked_targets = all_targets[sort_ind[::-1]]
            ranked_targets = np.logical_not(ranked_targets)
            sumed_arr = np.cumsum(ranked_targets)
            break_ind = np.where(sumed_arr == self.hn)[0][0]
            hncount = break_ind - (self.hn - 1)
            metrics["h" + str(self.hn)] = hncount / np.sum(all_targets)
        return metrics

    def reset(self):
        self.targets = []
        self.scores = []


class HNEvaluator(MaxEvaluator):
    def __init__(self, name, hn=50) -> None:
        super().__init__(name)
        self.targets = []
        self.scores = []
        self.hn = hn

    def collect_res(self, res, labels):
        self.scores.append(res.cpu().numpy())
        self.targets.append(labels.cpu().numpy())

    def summarize_res(self):
        metrics = {}
        all_scores = np.concatenate(self.scores).flatten()
        all_targets = np.concatenate(self.targets).flatten()
        is_labeled = np.logical_not(np.isnan(all_targets))
        all_scores = all_scores[is_labeled]
        all_targets = all_targets[is_labeled]
        sample_dist = np.sum(all_targets)
        if sample_dist == 0:
            metrics["apr"] = 0.0
            metrics["auc"] = 0.0
            metrics["h" + str(self.hn)] = 0.0
        elif sample_dist == len(all_targets):
            metrics["apr"] = 1.0
            metrics["auc"] = 1.0
            metrics["h" + str(self.hn)] = 1.0
        else:
            metrics["apr"] = met.average_precision_score(
                all_targets, all_scores
            )
            metrics["auc"] = met.roc_auc_score(all_targets, all_scores)
            # if self.hn is not None:
            if (
                self.hn is not None
                and len(all_targets) - sample_dist > self.hn
            ):
                sort_ind = np.argsort(all_scores)
                ranked_targets = all_targets[sort_ind[::-1]]
                ranked_targets = np.logical_not(ranked_targets)
                sumed_arr = np.cumsum(ranked_targets)
                break_ind = np.where(sumed_arr == self.hn)[0][0]
                hncount = break_ind - (self.hn - 1)
                metrics["h" + str(self.hn)] = hncount / np.sum(all_targets)
            else:
                metrics["h" + str(self.hn)] = 1.0
        return metrics

    def reset(self):
        self.targets = []
        self.scores = []


class BinaryAccEvaluator(MaxEvaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.targets = []
        self.scores = []

    def collect_res(self, res, labels):
        self.scores.append(res.cpu().numpy())
        self.targets.append(labels.cpu().numpy())

    def summarize_res(self):
        metrics = {}
        all_scores = np.concatenate(self.scores)
        all_targets = np.concatenate(self.targets)
        pos = np.argmax(all_scores, axis=-1)
        metrics["acc"] = met.accuracy_score(all_targets, pos)
        return metrics

    def reset(self):
        self.targets = []
        self.scores = []


class InfoNEEvaluator(MinEvaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.loss = []

    def collect_res(self, res):
        n = len(res)
        e_neg_mat = (
            res.view(-1)[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        )
        e_pos = torch.diagonal(res)
        loss = -torch.mean(
            torch.log(torch.exp(e_pos) / torch.exp(e_neg_mat).sum(dim=-1))
        )
        self.loss.append(loss.item())

    def summarize_res(self):
        metrics = {}
        metrics["mi_loss"] = np.array(self.loss).mean()
        return metrics

    def reset(self):
        self.loss = []


class LossEvaluator(MinEvaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.loss = []

    def collect_res(self, res):
        self.loss.append(res.item())

    def summarize_res(self):
        metrics = {}
        total_loss = np.array(self.loss).mean()
        metrics["loss"] = total_loss
        return metrics

    def reset(self):
        self.loss = []


class CollectionEvaluator(Evaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.out_val = []

    def collect_res(self, res):
        self.out_val.append(res)

    def summarize_res(self):
        metrics = {}
        res = torch.cat(self.out_val)
        metrics["res_col"] = res
        return metrics

    def reset(self):
        self.out_val = []


class MSEEvaluator(MinEvaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.pred = []
        self.labels = []

    def collect_res(self, res, labels):
        self.pred.append(res.cpu().numpy())
        self.labels.append(labels.cpu().numpy())

    def summarize_res(self):
        metrics = {}
        pred = np.concatenate(self.pred)
        labels = np.concatenate(self.labels)
        metrics["mse"] = met.mean_squared_error(labels, pred)
        return metrics

    def reset(self):
        self.pred = []
        self.labels = []
