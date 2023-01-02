import os
import torch
import argparse
import os.path as osp
import pickle as pkl
import json
from datetime import datetime
from datasets import HGVerGDGraphDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.utils.evaluators import BinaryAccEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.utils import (
    save_params,
    set_random_seed,
    k_fold2_split,
    k_fold_ind,
    hyperparameter_grid_search,
    multi_data_average_exp,
)
from gnnfree.nn.loss import MultiClassLoss
from gnnfree.utils.utils import write_res_to_file

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

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)

    with open(
        osp.join(
            params.data_folder_path, "graphs_Kary_Deterministic_Graphs.pkl"
        ),
        "rb",
    ) as file:
        d = pkl.load(file)

    graphs = []
    for g in d:
        row = g.row
        col = g.col
        goodind = row != col
        row = row[goodind]
        col = col[goodind]
        g = construct_graph_from_edges(row, col, n_entities=g.shape[0])
        g.ndata["feat"] = torch.ones([g.num_nodes(), 1])
        graphs.append(g)
    labels = torch.load(
        osp.join(params.data_folder_path, "y_Kary_Deterministic_Graphs.pt")
    ).numpy()

    params.num_tasks = 1
    params.num_class = int(labels.max() + 1)
    params.inp_dim = 1

    params.has_inp_feat = True

    fold = params.fold
    folds = k_fold_ind(labels, fold=fold)
    splits = k_fold2_split(folds, len(labels))
    data_col = []

    def construct_hg_ver_data(graphs, labels, ind):
        graph = [graphs[i] for i in ind]
        label = [labels[i] for i in ind]
        if not params.has_inp_feat:
            for g in graph:
                g.ndata["feat"] = torch.ones((g.num_nodes(), 1))
        d = HGVerGDGraphDataset(graph, label, params)
        return d

    for s in splits:
        train = construct_hg_ver_data(graphs, labels, s[0])
        test = construct_hg_ver_data(graphs, labels, s[1])
        val = construct_hg_ver_data(graphs, labels, s[2])
        data_col.append([train, test, val])

    # ---------------- Data Preparation End ------------- #

    evlter = BinaryAccEvaluator("acc_eval")
    eval_metric = "acc"

    loss = MultiClassLoss()

    def run_exp(data, params):
        gnn = HomogeneousGNN(params.reach_dist, params.inp_dim, params.emb_dim)
        graph_classifier = GDGraphClassifier(
            params.emb_dim, gnn, params.gd_deg
        )

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
            max_epochs=params.num_epochs,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor=eval_metric, mode="max"),
            ],
            logger=CSVLogger(save_dir=params.exp_dir),
        )
        trainer.fit(graph_pred)
        valid_res = trainer.validate()[0]
        test_res = trainer.test()[0]
        return valid_res, test_res

    hparams = (
        {"num_layers": [2, 3, 4, 5]}
        if params.psearch
        else {"num_layers": [params.num_layers]}
    )

    best_res = hyperparameter_grid_search(
        hparams,
        data_col,
        multi_data_average_exp,
        params,
        eval_metric,
        evlter,
        exp_arg=run_exp,
    )

    write_res_to_file(
        osp.join(params.exp_dir, "result"),
        params.train_data_set,
        eval_metric,
        best_res["test_mean"],
        best_res["val_mean"],
        best_res["test_std"],
        best_res["val_std"],
        best_res["params"],
        json.dumps(best_res),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gnn")

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="brazil_airport")

    parser.add_argument(
        "--emb_dim", type=int, default=32, help="overall embedding dimension"
    )
    parser.add_argument(
        "--mol_emb_dim",
        type=int,
        default=32,
        help="embedding dimension for atom/bond",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of GNN layers"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="jumping knowledge, should be 'last' or 'sum'",
    )

    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--l2", type=float, default=0, help="l2 regularization strength"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="evaluation batch size"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="number of workers in dataloading",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="number of epochs in one training routine",
    )

    parser.add_argument(
        "--reach_dist",
        type=int,
        default=3,
        help="max cutoff distance to find geodesics",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=10,
        help="number of fold for cross validation",
    )

    parser.add_argument(
        "--gd_type",
        type=str,
        default="HorGD",
        help="geodesic types, should be VerGD or HorGD",
    )

    parser.add_argument(
        "--gd_deg",
        type=bool,
        default=False,
        help="whether to use geodesic degrees for VerGD",
    )

    parser.add_argument(
        "--psearch",
        type=bool,
        default=False,
        help="perform hyperparameter search",
    )

    params = parser.parse_args()
    set_random_seed(1)
    main(params)
