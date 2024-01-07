import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class ToxicModel(pl.LightningModule):
    """
    A PyTorch Lightning module for the ToxicModel.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
        device (str, optional): Device to be used for training. Defaults to None.
        dtype (str, optional): Data type to be used for training. Defaults to None.

    Attributes:
        cfg (dict): Configuration dictionary containing model parameters.
        backbone (MambaLMHeadModel): Backbone model for the ToxicModel.
        mlp (nn.Sequential): Multi-layer perceptron for classification.
        criterion (nn.MSELoss): Mean squared error loss function.

    Methods:
        process_weights(weights): Process the weights dictionary.
        allocate_inference_cache(batch_size, max_seqlen, dtype=None, **kwargs): Allocate inference cache.
        forward(input_ids, slicer): Forward pass of the model.
        configure_optimizers(): Configure the optimizer and learning rate scheduler.
        calculate_accuracy(logits, label): Calculate the accuracy of the model.
        training_step(batch, batch_idx): Training step of the model.
        validation_step(batch, batch_idx): Validation step of the model.
    """    

    def __init__(
            self,
            cfg: dict,
            device=None,
            dtype=None,
        ) -> None:
            """
            Initializes the ToxicModel.

            Args:
                cfg (dict): Configuration dictionary containing model parameters.
                device (str, optional): Device to be used for model computations. Defaults to None.
                dtype (str, optional): Data type to be used for model computations. Defaults to None.
            """
            self.cfg = cfg
            
            mambaConf = MambaConfig(d_model=self.cfg.model.d_model,
                                    n_layer=self.cfg.model.n_layers,
                                   vocab_size=self.cfg.model.vocab_size)
            d_output = self.cfg.model.d_output
            factory_kwargs = {"device": device, "dtype": dtype}

            super().__init__()
            model = MambaLMHeadModel(mambaConf)
            model.load_state_dict(torch.load(self.cfg.model.weights_path))
            self.backbone = model.backbone
            if self.cfg.model.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            
            self.mlp = nn.Sequential(
                nn.Linear(self.cfg.model.d_model, 512),
                nn.ReLU(),
                nn.Dropout(self.cfg.model.dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(self.cfg.model.dropout),
                nn.Linear(256, d_output)
            )
            self.criterion = nn.MSELoss()
        
    def process_weights(self, weights):
        weights.pop('lm_head.weight')
        return weights
        

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, slicer):
        """
        Forward pass of the Mamba model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            slicer (torch.Tensor): Slicer tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        hidden_states = self.backbone(input_ids)
        hidden_states = hidden_states[torch.arange(slicer.size(0)), slicer-1]
        logits = self.mlp(hidden_states)
        return logits

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the model.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
                The optimizer is an instance of torch.optim.Adam with the learning rate
                specified by self.cfg.model.learning_rate. The learning rate scheduler
                is an instance of torch.optim.lr_scheduler.ReduceLROnPlateau with a
                patience of 3 and a cooldown of 1. The learning rate scheduler is
                monitored using the validation loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, cooldown=1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
        }
    
    def calculate_accuracy(self, logits, label):
        
        prediction = (torch.sigmoid(logits) >= 0.5).float()

        correct_predictions = torch.sum(prediction == label).item()
        accuracy = correct_predictions / (label.shape[0] * label.shape[1])

        return accuracy
    
    
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step for the Mamba model.

        Args:
            batch (tuple): A tuple containing the input tokens, slicer, and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value calculated during the validation step.
        """
        tokens = batch[0].squeeze(1)
        slicer = batch[1]
        labels = batch[2]
        logits = self(tokens, slicer).squeeze(1)
        loss = self.criterion(logits, labels)
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step on a batch of data.

        Args:
            batch (tuple): A tuple containing the input tokens, slicer, and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value calculated during the validation step.
        """
        tokens = batch[0].squeeze(1)
        slicer = batch[1]
        labels = batch[2]
        logits = self(tokens, slicer).squeeze(1)
        loss = self.criterion(logits, labels)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss
        