import torch
import dgl
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningModule
from gnnfree.nn.models.basic_models import MLPLayers, Predictor

"""Pytorch lightning pipeline for node, link, graph level tasks.
"""


class BaseTemplate(LightningModule):
    """Basic template, controls dataloading"""

    def __init__(self, save_dir, datasets, params):

        super().__init__()

        # Set our init args as class attributes
        self.save_dir = save_dir
        self.params = params
        self.datasets = datasets

        if not hasattr(self.params, "eval_sample_size"):
            self.params.eval_sample_size = None
        if not hasattr(self.params, "train_sample_size"):
            self.params.train_sample_size = None

    def creata_dataloader(
        self, data, size, batch_size, drop_last=True, shuffle=True
    ):
        if size is None:
            return DataLoader(
                data,
                batch_size=batch_size,
                num_workers=self.params.num_workers,
                collate_fn=data.get_collate_fn(),
                shuffle=shuffle,
                pin_memory=True,
                drop_last=drop_last,
            )
        else:
            return DataLoader(
                data,
                batch_size=batch_size,
                num_workers=self.params.num_workers,
                collate_fn=data.get_collate_fn(),
                sampler=RandomSampler(
                    data,
                    num_samples=size,
                    replacement=True,
                ),
                drop_last=drop_last,
            )

    def train_dataloader(self):
        return self.creata_dataloader(
            self.datasets["train"],
            self.params.train_sample_size,
            self.params.batch_size,
        )

    def val_dataloader(self):
        return self.creata_dataloader(
            self.datasets["val"],
            self.params.eval_sample_size,
            self.params.eval_batch_size,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.creata_dataloader(
            self.datasets["test"],
            self.params.eval_sample_size,
            self.params.eval_batch_size,
            drop_last=False,
            shuffle=False,
        )


class PredictorTemplate(BaseTemplate, metaclass=ABCMeta):
    """Template with a GNN encoder, a task_predictor that extract information
    from the embedding from GNN encoder. Then apply a mlp onto the extracted
    information.
    """

    def __init__(
        self,
        save_dir,
        datasets,
        params,
        gnn,
        task_predictor,
        loss,
        evlter,
        out_dim=1,
        b_norm=True,
    ):

        super().__init__(save_dir, datasets, params)

        # Set our init args as class attributes

        self.gnn = gnn
        self.task_predictor = task_predictor

        self.mlp = MLPLayers(
            3,
            [self.task_predictor.get_out_dim(), 512, 512, out_dim],
            batch_norm=b_norm,
        )

        self.model = Predictor(self.task_predictor, self.mlp)

        self.loss = loss

        self.evlter = evlter

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.l2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.5
                ),
                "monitor": "loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }

    @abstractmethod
    def compute_loss(self, output, batch):
        pass

    @abstractmethod
    def eval_step(self, output, batch):
        pass

    @abstractmethod
    def eval_epoch(self):
        pass

    @abstractmethod
    def eval_reset(self):
        pass


class LinkPredTemplate(PredictorTemplate):
    def forward(self, graph, head, tail, batch):
        x = self.model(graph, head, tail, batch)
        return x

    def on_train_epoch_start(self):
        # a new grpah with masked edge for every batch during training
        self.task_predictor.embedding_only_mode(False)

    def training_step(self, batch, batch_idx):
        g = self.datasets["train"].graph.to(batch.device)
        # graph edges masking
        edge_mask = batch.edge_mask
        edge_bool = torch.ones(
            g.num_edges(), dtype=torch.bool, device=batch.device
        )
        edge_bool[edge_mask] = 0
        subg = dgl.edge_subgraph(g, edge_bool, relabel_nodes=False)
        batch.g = subg

        score = self(batch.g, batch.head, batch.tail, batch)
        loss = self.compute_loss(score, batch)
        self.log(
            "loss",
            loss,
            batch_size=self.params.batch_size,
        )
        return loss

    def on_validation_epoch_start(self):
        # save a encoded graph with node representation for fast inference
        self.saved_g = self.datasets["train"].graph.to(self.device)
        self.saved_g.ndata["repr"] = self.gnn(self.saved_g)
        self.task_predictor.embedding_only_mode(True)

    def validation_step(self, batch, batch_idx):
        score = self(self.saved_g, batch.head, batch.tail, batch)
        loss = self.compute_loss(score, batch)
        self.eval_step(score, batch)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.params.batch_size,
        )
        return loss

    def on_validation_epoch_end(self):
        self.saved_g = None
        self.task_predictor.embedding_only_mode(False)
        res = self.eval_epoch()
        for k in res:
            self.log(k, res[k], on_step=False, on_epoch=True)
        self.eval_reset()

    def test_step(self, batch, batch_idx):
        score = self(self.saved_g, batch.head, batch.tail, batch)
        loss = self.compute_loss(score, batch)
        self.eval_step(score, batch)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.params.batch_size,
        )


class HGLinkPred(LinkPredTemplate):
    def compute_loss(self, output, batch):
        return self.loss(output, batch.labels)

    def eval_step(self, output, batch):
        self.evlter.collect_res(output, batch.labels)

    def eval_epoch(self):
        return self.evlter.summarize_res()

    def eval_reset(self):
        self.evlter.reset()


class KGLinkPred(LinkPredTemplate):
    def compute_loss(self, output, batch):
        return self.loss(output)

    def eval_step(self, output, batch):
        self.evlter.collect_res(output, batch.bsize)

    def eval_epoch(self):
        return self.evlter.summarize_res()

    def eval_reset(self):
        self.evlter.reset()


class KGFilteredLinkPred(KGLinkPred):
    def val_dataloader(self):
        return [
            self.creata_dataloader(
                self.datasets["val"][0],
                self.params.eval_sample_size,
                drop_last=False,
                shuffle=False,
            ),
            self.creata_dataloader(
                self.datasets["val"][1],
                self.params.eval_sample_size,
                drop_last=False,
                shuffle=False,
            ),
        ]

    def test_dataloader(self):
        return [
            self.creata_dataloader(
                self.datasets["test"][0],
                self.params.eval_sample_size,
                drop_last=False,
                shuffle=False,
            ),
            self.creata_dataloader(
                self.datasets["test"][1],
                self.params.eval_sample_size,
                drop_last=False,
                shuffle=False,
            ),
        ]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        score = self(self.saved_g, batch.head, batch.tail, batch)
        self.eval_step(score, batch)

    def test_step(self, batch, batch_idx, dataloader_idx):
        score = self(self.saved_g, batch.head, batch.tail, batch)
        self.eval_step(score, batch)


class GraphPredTemplate(PredictorTemplate):
    def forward(self, graph, batch):
        x = self.model(graph, batch)
        return x

    def training_step(self, batch, batch_idx):
        score = self(batch.g, batch)
        loss = self.compute_loss(score, batch)
        self.log(
            "loss",
            loss,
            batch_size=self.params.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        score = self(batch.g, batch)
        loss = self.compute_loss(score, batch)
        self.eval_step(score, batch)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.params.batch_size,
        )
        return loss

    def on_validation_epoch_end(self):
        res = self.eval_epoch()
        for k in res:
            self.log(k, res[k], on_step=False, on_epoch=True)
        self.eval_reset()

    def test_step(self, batch, batch_idx):
        score = self(batch.g, batch)
        loss = self.compute_loss(score, batch)
        self.eval_step(score, batch)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            batch_size=self.params.batch_size,
        )


class HGGraphPred(GraphPredTemplate):
    def compute_loss(self, output, batch):
        return self.loss(output, batch.labels)

    def eval_step(self, output, batch):
        self.evlter.collect_res(output, batch.labels)

    def eval_epoch(self):
        return self.evlter.summarize_res()

    def eval_reset(self):
        self.evlter.reset()


class NodePredTemplate(PredictorTemplate):
    def forward(self, graph, node, batch):
        x = self.model(graph, node, batch)
        return x

    def on_train_epoch_start(self):
        self.task_predictor.embedding_only_mode(False)

    def training_step(self, batch, batch_idx):
        g = self.datasets["train"].graph.to(batch.device)
        batch.g = g
        score = self(batch.g, batch.node, batch)
        loss = self.compute_loss(score, batch)
        self.log(
            "loss",
            loss,
            batch_size=self.params.batch_size,
        )
        return loss

    def on_validation_epoch_start(self):
        self.saved_g = self.datasets["train"].graph.to(self.device)
        self.saved_g.ndata["repr"] = self.gnn(self.saved_g)
        self.task_predictor.embedding_only_mode(True)

    def validation_step(self, batch, batch_idx):
        score = self(self.saved_g, batch.node, batch)
        loss = self.compute_loss(score, batch)
        self.eval_step(score, batch)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.params.batch_size,
        )
        return loss

    def on_validation_epoch_end(self):
        self.saved_g = None
        self.task_predictor.embedding_only_mode(False)
        res = self.eval_epoch()
        for k in res:
            self.log(k, res[k], on_step=False, on_epoch=True)
        self.eval_reset()

    def test_step(self, batch, batch_idx):
        score = self(self.saved_g, batch.node, batch)
        loss = self.compute_loss(score, batch)
        self.eval_step(score, batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=self.params.batch_size,
        )


class HGNodePred(NodePredTemplate):
    def compute_loss(self, output, batch):
        return self.loss(output, batch.labels)

    def eval_step(self, output, batch):
        self.evlter.collect_res(output, batch.labels)

    def eval_epoch(self):
        return self.evlter.summarize_res()

    def eval_reset(self):
        self.evlter.reset()
