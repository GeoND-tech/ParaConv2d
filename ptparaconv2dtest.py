import torch
import geondptfree as gpt
import ptparaconv2d

x = torch.rand(1, 3, 5, 5)
parameters = torch.rand(1, 2*28).cpu().numpy()
pb1 = gpt.ParaConv2d(3, 1, kernel_size=3).cpu()
pb2 = ptparaconv2d.ParaConv2d(3, 1, kernel_size=3, init_from_numpy = torch.clone(pb1.pars).detach().numpy()).cpu()

print("CPU")

print("geondptfree")
print(pb1(x))
print("ptparaconv2d")
print(pb2(x))

print("CUDA")

pb1.cuda()
pb2.cuda()
print("geondptfree")
print(pb1(x.cuda()))
print("ptparaconv2d")
print(pb2(x.cuda()))

