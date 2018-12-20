# CenterLoss_pytorch
Pytorch CenterLoss

## Example
```python
import torch
from center_loss import CenterLoss
centerloss = CenterLoss(128, 10)
input = torch.randn(3, 128)
target = torch.LongTensor([3, 6, 2])
loss = centerloss(input, target)
print loss
```
