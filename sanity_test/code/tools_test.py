import tools
import torch

a = torch.randn(1,6).cuda()
b = tools.stereographic_project(a)
c = tools.stereographic_unproject(b)

print (tools.normalize_vector(a))
print (tools.normalize_vector(b))
print (tools.normalize_vector(c))