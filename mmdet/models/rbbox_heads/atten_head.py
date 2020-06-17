import torch.nn as nn
from ..registry import HEADS

@HEADS.register_module
class AttenHead(nn.Module):
    def __init__(self,
                in_channels=1024,
                atten_size=14):
        super(AttenHead, self).__init__()
        self.atten_size = atten_size
        self.atten = nn.Linear(in_channels, atten_size**2)
    
    def init_weights(self):
        nn.init.normal_(self.atten.weight, 0, 0.01)
        nn.init.constant_(self.atten.bias, 0)
    
    def forward(self, x):
        x = self.atten(x)
        x = x.view(x.shape[0], self.atten_size, self.atten_size)
        return x