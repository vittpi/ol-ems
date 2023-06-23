from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.plots import total_cost

import logging
logging = logging.getLogger('pytorch_lightning')


class MSELoss(nn.Module):
    """This class is used to create a mean squared error loss function."""

    def __init__(self) -> None:
        super(MSELoss, self).__init__()
        self.loss = F.mse_loss

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """This method is used to perform a forward pass through the loss function.
        
        Args:
            y_hat (torch.Tensor): The predicted values.
            y (torch.Tensor): The true values.
            weight (Optional[torch.Tensor], optional): The weight vector. Defaults to None.
        """
        batch_size, _, _ = y_hat.shape
        loss = 0.0
        for i in range(batch_size):
            loss_per_sample = self.loss(y_hat[i], y[i], reduction='none')
            if weight is not None:
                loss_per_sample = loss_per_sample * weight[:, None]
            loss += loss_per_sample.sum()
        loss = loss / batch_size
        return loss

    def get_weight(self, prediction_horizon: int, exp_decay: float,
                   device: torch.device) -> torch.Tensor:
        """Create a weight vector for the loss function, different for each prediction horizon."""
        return exp_decay ** torch.range(0, prediction_horizon-1, device=device)


class TotalCostLoss(nn.Module):

    def __init__(self) -> None:
        super(TotalCostLoss, self).__init__()

    def forward(self, price: torch.Tensor, Pg: torch.Tensor) -> torch.Tensor:
        """This method is used to perform a forward pass through the loss function.
        
        Args:
            price (torch.Tensor): The price vector.
            Pg (torch.Tensor): The power exchanged with the grid vector.
        """
        return total_cost(price=price, Pg=Pg)
