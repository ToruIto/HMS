
import torch
import timm
import torch.nn as nn

class HMSHBACSpecModel(nn.Module):
    def __init__(self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            feature_extractor: nn.Module,
        ):
        super().__init__()
        self.feature_extractor=feature_extractor
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels)

    def forward(self, x):
        x = torch.permute(x,(0,2,1))
        x = self.feature_extractor(x)
        h = self.model(x)

        return h