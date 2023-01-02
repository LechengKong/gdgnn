import os
import torch
import argparse
import os.path as osp
import json
from datetime import datetime
from datasets import HGVerGDGraphDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.utils.evaluators import BinaryHNEvaluator
from gnnfree.utils.utils import (
    save_params,
    set_random_seed,
    hyperparameter_grid_search,
    write_res_to_file,
)
from gnnfree.nn.loss import BinaryLoss

from ogb.graphproppred import DglGraphPropPredDataset

from models.GNN import MOLGNNVN, MOLGNN
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

    gnn_model = MOLGNNVN if params.train_data_set == "ogbg-molpcba" else MOLGNN

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)

    data = DglGraphPropPredDataset(
        name=params.train_data_set, root=params.data_path
    )
    split_idx = data.get_idx_split()
    train_ind = split_idx["train"]
    test_ind = split_idx["test"]
    val_ind = split_idx["valid"]
    train_graph = [data[i][0] for i in train_ind]
    test_graph = [data[i][0] for i in test_ind]
    val_graph = [data[i][0] for i in val_ind]

    train_label = [data[i][1].numpy() for i in train_ind]
    test_label = [data[i][1].numpy() for i in test_ind]
    val_label = [data[i][1].numpy() for i in val_ind]

    params.num_class = len(train_label[0])
    params.atom_dim = train_graph[0].ndata["feat"]
    params.bond_dim = train_graph[0].edata["feat"]
    params.inp_dim = 0

    def construct_hg_ver_data(data, label_t):
        graph = data
        for g in graph:
            g.ndata["atom_feat"] = g.ndata["feat"]
            g.edata["bond_feat"] = g.edata["feat"]
            g.ndata["feat"] = torch.zeros([g.num_nodes(), 0])
            g.edata["feat"] = torch.zeros([g.num_edges(), 0])
        # d = HGNeighborGraphDataset(graph, label_t, params)
        d = HGVerGDGraphDataset(graph, label_t, params)
        return d

    data_col = [
        construct_hg_ver_data(train_graph, train_label),
        construct_hg_ver_data(test_graph, test_label),
        construct_hg_ver_data(val_graph, val_label),
    ]

    # -------------------- Data Preparation ---------------- #

    evlter = BinaryHNEvaluator("acc_eval")
    if params.train_data_set == "ogbg-molpcba":
        eval_metric = "apr"
    else:
        eval_metric = "auc"

    loss = BinaryLoss()

    def run_exp(data, params):
        gnn = gnn_model(
            params.reach_dist,
            params.inp_dim,
            params.emb_dim,
            params.mol_emb_dim,
            0,
        )
        graph_classifier = GDGraphClassifier(
            params.emb_dim, gnn, params.gd_deg
        )

        graph_pred = HGGraphPred(
            params.exp_dir,
            {
                "train": data_col[0],
                "test": data_col[1],
                "val": data_col[2],
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
        valid_res = trainer.validate()[0]
        test_res = trainer.test()[0]
        return valid_res, test_res

    hparams = (
        {"num_layers": [2, 3, 4, 5]}
        if params.psearch
        else {"num_layers": [params.num_layers]}
    )

    best_res = hyperparameter_grid_search(
        hparams, data_col, run_exp, params, eval_metric, evlter
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
