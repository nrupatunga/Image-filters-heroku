"""

File: filter_trainer.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: trainer code
"""

import pytorch_lightning as pl

from custom_nets import FIP


class LitModel(pl.LightningModule):

    """Docstring for LitModel. """

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int = 6,
                 lr: float = 1e-4,
                 **kwargs) -> None:
        """
        @lr: learning rate
        """
        super().__init__()

        self.model = FIP()
        self.save_hyperparameters()

    def forward(self, x):
        """forward function for litmodel
        """
        return self.model(x)
