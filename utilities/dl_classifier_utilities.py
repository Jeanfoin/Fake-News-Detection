from .plot_utilities import (
    figure_to_tensor,
    plot_confusion_matrix,
    plot_full_roc_auc
)

from typing import Dict, Any
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import time
import lightning as L

class TextNewsModule(L.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating a neural network on UTKFace dataset.

    Args:
        neural_net (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer: The optimizer used for training.
        model_name (str, optional): The name of the model. Default is 'my_model'.
        scheduler (optional): Learning rate scheduler.
        monitor (Dict, optional): A dictionary containing metric and mode to monitor during training.
    """

    def __init__(
        self,
        neural_net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer,
        model_name: str = "my_model",
        scheduler=None,
        monitor: Dict = {"metric": "val_loss", "mode": "min"},
    ):
        """
        Initializes the UTKFaceModule with the provided model, loss function, optimizer, etc.

        Args:
            neural_net (torch.nn.Module): The model to use for training.
            criterion (torch.nn.Module): The loss function for training.
            optimizer: The optimizer for training.
            model_name (str, optional): The name of the model.
            scheduler (optional): The learning rate scheduler.
            monitor (Dict, optional): Dictionary specifying the metric and mode to monitor during training.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["neural_net", "criterion", "optimizer", "scheduler"]
        )
        self.neural_net = neural_net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor
        self.model_name = model_name

    def setup(self, stage=None):
        """
        Setup method to initialize metric tracking for train, validation, and test stages.

        Args:
            stage (str, optional): The stage of the model (train, val, test).
        """
        self.train_metrics = []
        self.last_train_metrics = {}
        self.val_metrics = []
        self.last_val_metrics = {}
        self.test_metrics = []
        self.last_test_metrics = {}
        self.test_inference_times = []


    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model, returns the loss and logits.

        Args:
            img (torch.Tensor): The input image tensor.
            labels (torch.Tensor, optional): The ground truth labels. If None, returns zero loss.

        Returns:
            tuple: A tuple containing loss and logits.
        """
        logits = self.neural_net(input_ids=input_ids, attention_mask=attention_mask)
        
        if labels is None:
            loss = torch.tensor(0.0, device=text.device, dtype=torch.float32)
        else:
            labels = labels.unsqueeze(1).float()
            loss = self.criterion(logits, labels)
        return loss, logits

    def _common_step(self, batch, stage):
        """
        A common step used for training, validation, or testing.

        Args:
            batch (dict): A batch of data including images and metadata.
            stage (str): The current stage of the model ('train', 'val', 'test').

        Returns:
            dict: A dictionary containing the loss, logits, and labels.
        """

        ids = batch["ids"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        texts = batch["news_text"]
        labels = batch["labels"]
        
        loss, logits = self(input_ids, attention_mask, labels)

        log_kwargs = {
            "prog_bar": True,
            "logger": True,
            "on_step": stage == "train",
            "on_epoch": stage != "train",
        }

        dict_log = {f"{stage}_loss": loss}
        self.log_dict(dict_log, **log_kwargs)

        return {"loss": loss, "ids": ids, "logits": logits, "labels": labels}

    def on_train_start(self):
        """
        Records the start time of training.
        """
        self.train_start_time = time.time()

    def on_train_end(self):
        """
        Records the end time of training and calculates the total training time.
        """
        self.training_time = time.time() - self.train_start_time

    def training_step(self, batch, batch_idx):
        """
        Performs a training step and logs the loss.

        Args:
            batch (dict): A batch of data containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        metrics = self._common_step(batch, "train")
        self.train_metrics.append(metrics)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step and logs the validation loss and metrics.

        Args:
            batch (dict): A batch of data containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: The validation metrics.
        """
        metrics = self._common_step(batch, "val")
        self.val_metrics.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        """
        Performs a test step and logs the test loss and metrics, along with inference times.

        Args:
            batch (dict): A batch of data containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: The test metrics.
        """
        start_time = time.time()
        with torch.no_grad():
            metrics = self._common_step(batch, "test")
        inference_time = time.time() - start_time

        self.test_inference_times.append(inference_time)
        self.test_metrics.append(metrics)
        return metrics

    def _compute_metrics(self, outputs, stage):
        """
        Computes metrics based on model predictions and ground truth labels for a given stage.

        Args:
            outputs (list): A list of outputs containing loss, logits, and labels.
            stage (str): The stage for which the metrics are computed ('train', 'val', 'test').

        Returns:
            tuple: A tuple containing labels, logits, and computed metrics.
        """

        ids = torch.cat([x["ids"].detach().cpu() for x in outputs])
        labels = torch.cat([x["labels"].detach().cpu() for x in outputs])
        logits = torch.cat([x["logits"].detach().cpu() for x in outputs])
        probas = torch.sigmoid(logits).squeeze(-1)
        predictions = (probas > 0.5).int()

        metrics = {}
        f1 = f1_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        roc_auc = roc_auc_score(labels, probas)

        metrics["f1_score"] = f1
        metrics["roc_auc_score"] = roc_auc
        metrics["accuracy_score"] = accuracy

        for name, value in metrics.items():
            self.logger.experiment.add_scalar(
                f"{stage}_{name}", value, self.current_epoch
            )

        return {"ids": ids, 
                "labels": labels, 
                "logits": logits, 
                "probas": probas, 
                "predictions": predictions, 
                "metrics": metrics}

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch. Computes and logs training metrics.
        """
        metrics = self._compute_metrics(self.train_metrics, "train")
        self.last_train_metrics = metrics["metrics"]
        self.train_metrics.clear()

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch. Computes and logs validation metrics.
        """
        metrics = self._compute_metrics(self.val_metrics, "val")
        self.last_val_metrics = metrics["metrics"]
        for name, value in metrics["metrics"].items():
            self.log(
                f"val_{name}",
                value,
                prog_bar=(
                    True if name in [self.monitor["metric"], "val_loss"] else False
                ),
            )
        self.val_metrics.clear()

    def _log_inference_stats(self):
        """
        Logs statistics about the inference times (mean, std, median) to TensorBoard.
        """
        times = np.array(self.test_inference_times)
        stats = {
            "mean": np.mean(times),
            "std": np.std(times),
            "median": np.median(times),
        }
        self.last_test_inference_stats = stats
        for name, value in stats.items():
            self.log(f"test_inference_time_{name}", value)

        self.logger.experiment.add_histogram(
            "test_inference_time_distribution", torch.tensor(times), self.current_epoch
        )

    def _log_roc_curves(self, labels, probas):
        """
        Logs ROC curves for classification tasks to TensorBoard.

        Args:
            labels (dict): Ground truth labels.
            logits (dict): Model predictions.
        """

        fig, _ = plot_full_roc_auc(probas, labels, num_classes=2)
        im = figure_to_tensor(fig)
        self.logger.experiment.add_image(
            f"test_roc_curves",
            im,
            global_step=self.current_epoch,
        )

    
    def _log_classification_report(self, labels, predictions):
        """
        Logs classification reports for each classification task to TensorBoard.

        Args:
            labels (dict): Ground truth labels.
            logits (dict): Model predictions.
        """

        report = classification_report(
            labels,
            predictions,
            output_dict=False,
            zero_division=np.nan
        )
        self.logger.experiment.add_text(
            f"test_classification_report", report
        )

    

    def _log_confusion_matrix(self, labels, predictions):
        """
        Logs confusion matrices for each classification task to TensorBoard.

        Args:
            labels (dict): Ground truth labels.
            logits (dict): Model predictions.
        """

        cm = confusion_matrix(labels, predictions)
        fig, _ = plot_confusion_matrix(cm)
        im = figure_to_tensor(fig)
        self.logger.experiment.add_image(
            f"test_confusion_matrix",
            im,
            global_step=self.current_epoch,
        )

    def on_test_epoch_end(self):
        """
        Called at the end of each test epoch. Computes and logs test metrics.
        """
        metrics = self._compute_metrics(self.test_metrics, "test")
        self.last_test_metrics = metrics["metrics"]
        for name, value in metrics["metrics"].items():
            self.log(
                f"test_{name}",
                value,
                prog_bar=(
                    True if name in [self.monitor["metric"], "val_loss"] else False
                ),
            )
        self.test_metrics.clear()


        results_df = pd.DataFrame({"ids": metrics["ids"], 
                                   "probas": metrics["probas"], 
                                   "predictions": metrics["predictions"], 
                                   "labels": metrics["labels"]})
        
        # Save to CSV
        csv_path = f"../logs/test_results_{self.model_name}.csv"
        self.test_data = csv_path
        results_df.to_csv(csv_path, index=False)

        self._log_inference_stats()
        self._log_confusion_matrix(metrics["labels"], metrics["predictions"])
        self._log_roc_curves(metrics["labels"], metrics["probas"])
        self._log_classification_report(metrics["labels"], metrics["predictions"])
        self.test_metrics.clear()
        self.test_inference_times.clear()
        
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for training.

        Returns:
            dict: Dictionary containing the optimizer and learning rate scheduler if available.

        Raises:
            ValueError: If no optimizer is provided.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be provided")
        if self.scheduler is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}
        }
