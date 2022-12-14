import os
import torch
import argparse
import os.path as osp
import dgl
from datetime import datetime
from datasets import HGVerGDGraphDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.utils.evaluators import BinaryAccEvaluator
from gnnfree.utils.utils import (
    save_params,
    set_random_seed,
    k_fold2_split,
    k_fold_ind,
)
from gnnfree.nn.loss import MultiClassLoss

from gnnfree.nn.models.GNN import HomogeneousGNN

from models.graph_classifier import GDGraphClassifier

from lightning_template import HGGraphPred


def main(params):
    # -------------------- Prepare Exp -------------------- #
    params = parser.parse_args()
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    params.exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir, "command"), params)
    params.prefix = params.train_data_set + "_"
    params.model_name = osp.join(params.exp_dir, params.prefix)
    # -------------------- Data Preparation ---------------- #

    data = dgl.data.TUDataset(params.train_data_set)

    params.num_tasks = len(data[0][1])
    params.has_inp_feat = "feat" in data[0][0].ndata
    if params.has_inp_feat:
        params.inp_dim = data[0][0].ndata["feat"].size()[1]
    else:
        params.inp_dim = 1

    c = 0
    for g, l in data:
        # print(l)
        c = max(c, l)
    params.num_class = c + 1

    labels = [val[1].item() for val in data]

    fold = params.fold
    folds = k_fold_ind(labels, fold=fold)
    splits = k_fold2_split(folds, len(labels))
    data_col = []

    def construct_hg_ver_data(data, labels, ind):
        graph = [data[i][0] for i in ind]
        label = [labels[i] for i in ind]
        if not params.has_inp_feat:
            for g in graph:
                g.ndata["feat"] = torch.ones((g.num_nodes(), 1))
        d = HGVerGDGraphDataset(graph, label, params)
        return d

    for s in splits:
        train = construct_hg_ver_data(data, labels, s[0])
        test = construct_hg_ver_data(data, labels, s[1])
        val = construct_hg_ver_data(data, labels, s[2])
        data_col.append([train, test, val])

    # ---------------- Data Preparation End ------------- #

    gnn = HomogeneousGNN(params.reach_dist, params.inp_dim, params.emb_dim)
    graph_classifier = GDGraphClassifier(params.emb_dim, gnn, params.gd_deg)

    evlter = BinaryAccEvaluator("acc_eval")
    eval_metric = "acc"

    loss = MultiClassLoss()

    graph_pred = HGGraphPred(
        params.exp_dir,
        {
            "train": data_col[0][0],
            "test": data_col[0][1],
            "val": data_col[0][2],
        },
        params,
        gnn,
        graph_classifier,
        loss,
        evlter,
        out_dim=params.num_class,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=3,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(monitor=eval_metric, mode="max"),
        ],
        logger=CSVLogger(save_dir=params.exp_dir),
    )
    trainer.fit(graph_pred)

    trainer.test()

    # best_res = hyperparameter_grid_search(
    #     hparams,
    #     data_col,
    #     multi_data_average_exp,
    #     params,
    #     eval_metric,
    #     evlter,
    #     exp_arg=run_exp,
    # )

    # write_res_to_file(
    #     osp.join(params.exp_dir, "result"),
    #     params.train_data_set,
    #     eval_metric,
    #     best_res["test_mean"],
    #     best_res["val_mean"],
    #     best_res["test_std"],
    #     best_res["val_std"],
    #     best_res["params"],
    #     json.dumps(best_res),
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gnn")

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="DD")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--JK", type=str, default="sum")
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--reach_dist", type=int, default=3)

    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument("--gdgnn", type=bool, default=True)
    parser.add_argument("--gd_deg", type=bool, default=True)

    parser.add_argument("--psearch", type=bool, default=False)

    params = parser.parse_args()
    set_random_seed(1)
    main(params)
