import torch

a = torch.rand(16, 2, 14, 1000)
print(a.shape)
b = torch.cat([a[:, 0, :, :], a[:, 1, :, :]], dim=1)
c = b.transpose(1,2).contiguous()
print(c.shape)