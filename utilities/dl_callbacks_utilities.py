import os
from typing import Dict, Any
from datetime import datetime
import time
import numpy as np
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger


class ModelTracker(L.Callback):
    """
    A callback to track model training and evaluation metrics, and save the results into a CSV file.

    Parameterss:
        csv_path (str, optional): Path to the CSV file where the results will be logged. Default is "../model_selection_log.csv".
    """

    def __init__(self, csv_path: str = "../model_selection_log.csv"):
    
        super().__init__()
        self.csv_path = csv_path
        self.val_metrics = {}
        self.test_metrics = {}
        self.start_time = datetime.now()
        self.best_model_path = None

    def get_model_id(self, pl_module) -> Dict[str, str]:
        """
        Retrieves the model's unique identifier.
    
        Parameters:
            pl_module (pl.LightningModule): The PyTorch Lightning model.
    
        Returns:
            dict: A dictionary containing the model's unique identifier.
        """
        return {"model_id": pl_module.model_name}

    def get_backbone_info(self, pl_module) -> Dict[str, Any]:
        """
        Retrieves information about the model's backbone (e.g., architecture and parameters).

        Parameters:
            pl_module (pl.LightningModule): The PyTorch Lightning model.

        Returns:
            dict: A dictionary containing the backbone name and the number of parameters in the backbone.
        """
        return {"num_params": sum(p.numel() for p in pl_module.neural_net.parameters())}

    def get_criterion_info(self, pl_module) -> Dict[str, Any]:
        """
        Retrieves information about the model's criterion (loss function).
    
        Parameters:
            pl_module (pl.LightningModule): The PyTorch Lightning model.
    
        Returns:
            dict: A dictionary containing the criterion's name and the number of parameters.
        """
        return {
            "general_criterion": pl_module.criterion.__class__.__name__,
            "general_criterion_params": sum(p.numel() for p in pl_module.criterion.parameters())}

    def get_training_info(self, pl_module, trainer) -> Dict[str, Any]:
        """
        Retrieves general training-related information.

        Parameters:
            pl_module (pl.LightningModule): The PyTorch Lightning model.
            trainer (pl.Trainer): The PyTorch Lightning trainer.

        Returns:
            dict: A dictionary containing the number of epochs, training time, batch size, and dataset sizes.
        """
        info = {}
        info["number_epochs"] = trainer.current_epoch
        info["training_time"] = pl_module.training_time
        info["batch_size"] = trainer.datamodule.batch_size
        info["train_set_len"] = len(trainer.datamodule.train_dataset)
        info["val_set_len"] = len(trainer.datamodule.val_dataset)
        return info

    def get_optimizer_info(self, pl_module) -> Dict[str, Any]:
        """
        Retrieves information about the model's optimizer.

        Parameters:
            pl_module (pl.LightningModule): The PyTorch Lightning model.

        Returns:
            dict: A dictionary containing the optimizer's name, learning rate, and weight decay.
        """
        """Extract optimizer information"""
        return {
            "optimizer_name": pl_module.optimizer.__class__.__name__,
            "lr": pl_module.optimizer.defaults["lr"],
            "weight_decay": pl_module.optimizer.defaults["weight_decay"],
        }

    def get_scheduler_info(self, pl_module, trainer) -> Dict[str, Any]:
        """
        Retrieves information about the model's learning rate scheduler.

        Parameters:
            pl_module (pl.LightningModule): The PyTorch Lightning model.
            trainer (pl.Trainer): The PyTorch Lightning trainer.

        Returns:
            dict: A dictionary containing the scheduler's name and the last learning rate.
        """
        info = {}
        if pl_module.scheduler is not None:
            info["scheduler_name"] = (pl_module.scheduler.__class__.__name__,)
            info["last_lr"] = pl_module.scheduler.get_last_lr()[0]

        return info

    def get_checkpoint_info(self, trainer) -> Dict[str, Any]:
        """
        Retrieves information about the model's checkpoint settings.
    
        Parameters:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
    
        Returns:
            dict: A dictionary containing the checkpoint directory and filename.
        """
        info = {}
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                info["checkpoint_dirpath"] = (callback.dirpath,)
                info["checkpoint_filename"] = callback.filename
        return info
    
    def get_logger_info(self, trainer) -> Dict[str, Any]:
        """
        Retrieves information about the logger used during training.
    
        Parameters:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
    
        Returns:
            dict: A dictionary containing the logger's directory and name.
        """
        info = {}
        for callback in trainer.callbacks:
            if isinstance(callback, TensorBoardLogger):
                info["log_dir"] = (trainer.logger.log_dir,)
                info["name"] = trainer.logger.name
        return info
    
    def on_validation_end(self, trainer, pl_module):
        """
        Callback method that is called at the end of the validation phase. It collects the validation metrics.
    
        Parameters:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
            pl_module (pl.LightningModule): The PyTorch Lightning model.
        """
        self.val_metrics = {
            f"val_{k}": v for k, v in pl_module.last_val_metrics.items()
        }
    
    def on_test_end(self, trainer, pl_module):
        """
        Callback method that is called at the end of the test phase. It collects the test metrics and inference times,
        and logs the results to a CSV file.
    
        Parameters:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
            pl_module (pl.LightningModule): The PyTorch Lightning model.
        """
        self.test_metrics = {
            f"test_{k}": v for k, v in pl_module.last_test_metrics.items()
        }
        inference_times = {
            f"test_{k}_inference_time": v
            for k, v in pl_module.last_test_inference_stats.items()
        }
    
        model_id = self.get_model_id(pl_module)
        backbone_info = self.get_backbone_info(pl_module)
        criterion_info = self.get_criterion_info(pl_module)
        training_info = self.get_training_info(pl_module, trainer)
        optimizer_info = self.get_optimizer_info(pl_module)
        scheduler_info = self.get_scheduler_info(pl_module, trainer)
        ckpt_info = self.get_checkpoint_info(trainer)
        logger_info = self.get_logger_info(trainer)
    
        run_info = {
            **model_id,
            **backbone_info,
            **criterion_info,
            **training_info,
            **optimizer_info,
            **scheduler_info,
            **ckpt_info,
            **logger_info,
            **self.val_metrics,
            **self.test_metrics,
            **inference_times,
        }
    
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                new_df = pd.DataFrame([run_info])
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame([run_info])
    
            df.to_csv(self.csv_path, index=False)
            print(f"Results successfully saved to {self.csv_path}")
    
        except Exception as e:
            print(f"Error saving results to CSV: {str(e)}")
            backup_path = (
                f"backup_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            pd.DataFrame([run_info]).to_csv(backup_path, index=False)
            print(f"Backup results saved to {backup_path}")

