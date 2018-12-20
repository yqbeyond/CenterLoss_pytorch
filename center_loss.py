import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, x, labels):
        """
        x: [batch_size, feature_len]
        labels: [bacth_size]
        """
        loss = 0.0
        for i, g in enumerate(labels):
            loss += torch.norm(x[i] - self.centers[g]) ** 2
        loss = loss / len(x)
        return loss
