import torch

'''x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)

z = x * x * y'''

'''z.backward()
print("[1]:", x.grad, y.grad)
# 12.0 4.0'''

'''grad_x = torch.autograd.grad(outputs=z, inputs=x)
print("[2]:", grad_x, grad_x[0])
# [2]: (tensor(12.),) tensor(12.)'''

'''grad_x = torch.autograd.grad(outputs=z, inputs=x, retain_graph=True)
grad_y = torch.autograd.grad(outputs=z, inputs=y)
print("[3]:", grad_x[0], grad_y[0])
# [3]: tensor(12.) tensor(4.)'''

'''grad_x = torch.autograd.grad(outputs=z, inputs=x, create_graph=True)
grad_x_x = torch.autograd.grad(outputs=grad_x, inputs=x)
print("[4]:", grad_x_x[0])
# [4]: tensor(6.)'''

'''grad_x = torch.autograd.grad(outputs=z, inputs=x, create_graph=True)
grad_x[0].backward()
print("[5]:", grad_x, x.grad)
# [5]: (tensor(12., grad_fn=<AddBackward0>),) tensor(6.)'''

'''z.backward(create_graph=True)
x.grad.backward()
print("[6]:", x.grad)
# [6]: tensor(24., grad_fn=<AddBackward0>)'''

'''z.backward(create_graph=True)
x.grad.data.zero_()
x.grad.backward()
print("[7]:", x.grad)
# [7]: tensor(6., grad_fn=<CopyBackwards>)'''

'''x = torch.tensor([1., 2.]).requires_grad_()
y = x * x

y.sum().backward()
print(x.grad)'''


import torch

a = torch.randn((3,3), requires_grad = True)

w1 = torch.randn((3,3), requires_grad = True)
w2 = torch.randn((3,3), requires_grad = True)
w3 = torch.randn((3,3), requires_grad = True)
w4 = torch.randn((3,3), requires_grad = True)

b = w1*a
c = w2*a

d = w3*b + w4*c

L = 10 - d

print("The grad fn for a is", a.grad_fn)
print("The grad fn for d is", d.grad_fn)
