import torch

x = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float32)
print("x:", x)

y = x + 2
print("y:", y)

z = y * 2
print("z:", z)
# PyTorch requires .backward() to be called on a scalar (0-dim tensor).
# - if not: RuntimeError: grad can be implicitly created only for scalar outputs
# because z is a vector (shape (3,)).  
# PyTorch doesn’t know how to propagate a single gradient through multiple outputs unless you give it extra info.


g = z.mean() # is had to be scalr value in last opreation else it has to be the vector it self 
print("g (mean):", g)

# First backward pass, 
g.backward(retain_graph=True) # dg/dx
print("x.grad after g.backward():", x.grad)

# Clear gradients before doing another backward pass
x.grad.zero_()

# Backward with vector v (Jacobian-vector product)
v = z
z.backward(v) 
print("x.grad after z.backward(v):", x.grad)


# - ------------------------------- previding grad in certain steps in trainnig 
# 1. x.requires_grad_(False) 
# 2. x.detach()
# 3 with torch.no_grad():
x.requires_grad_(False)

x.detach()

with torch.no_grad():

    y = x.detach()


weight = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weight * 3).sum()
    model_output.backward()

    print(weight.grad) 
    # missed freeing the grad once the loop ends

    weight.grad.zero_()


weight = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD([weight], lr=0.01) 

for epoch in range(3):
    optimizer.zero_grad()                

    model_output = (weight * 3).sum()  
    model_output.backward()            

    print(weight.grad)

    optimizer.step()                    