import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import RegNet_Y_400MF_Weights


class RegNet6(nn.Module):
    """
    RegNet-Y-400MF adapted for 6-channel input (RGB pair concatenation).
    Output: sigmoid probability of match.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        self.model = models.regnet_y_400mf(weights=weights)
        self.model.stem[0] = nn.Conv2d(6, 32, kernel_size=3, stride=2,
                                        padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class RegNet2(nn.Module):
    """
    RegNet-Y-400MF adapted for 2-channel input (grayscale pair concatenation).
    Output: sigmoid probability of match.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        self.model = models.regnet_y_400mf(weights=weights)
        self.model.stem[0] = nn.Conv2d(2, 32, kernel_size=3, stride=2,
                                        padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def load_model(model_class, path: str, device: torch.device):
    """Loads saved weights into a model instance."""
    m = model_class(pretrained=False).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m
