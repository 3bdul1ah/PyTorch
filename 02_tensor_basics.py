import torch 

x = torch.empty(1) # .rand(), .zeros() (1d "elements in one row") or (2d  (2, 2))
# by defualt dtype is float32 x.dtype or u can add it in the x args (2,2, dtype=torch.TYPE)
x = torch.ones(1, dtype=torch.float16)

print(x)

y = torch.tensor([2, 1.5])
print(y)

# element add
# [[1 + 2.0, 1 + 1.5],
#  [1 + 2.0, 1 + 1.5]]
z = x + y
print(z)

y.add_(x)
print(y) # == z but is modifying y directly using inplace opreator _

g = x - y 
g  = torch.sub(x,y) # .mul() or y.mul_(x) .div()
print(g)

print("##### --- ####")

x = torch.ones(5, 3)
print(x[:, 1]) # all rows but only 2nd colm [MAT]^ T


x = torch.rand(4, 4)
print(x)
# reshaping
y = x.view(16) # if u want to determine other size use -1 then tourch will find suitable size
print(y)

c =  x.view(-1, 8)
print(c)

# converting to nmpuy to tesnor or vise versa

import numpy 

a = torch.ones(5)
print(type(a))

b = a.numpy()
print(type(b))


c = torch.from_numpy(b)
print(type(c))

# if tensor are in CPU not GPU:
# it will add +1 to both as it share same memoery pointer for both object 
a.add_(1)
print(a)
print(b)

# GPU:
a_gpu = torch.ones(5, device="cuda")
b = a_gpu.cpu().numpy() #.cpu() to copy the tensor to host memory first.

a_gpu.add_(1)
print(a_gpu, b)

# another way
if torch.cuda.is_available():
    device = torch.device("cuda")

    x = torch.ones(5, device=device)
    y = torch.ones(5) # tensor not on cuda 
    # to mvoe it to cuda will be:
    y = y.to(device) 
    z = x + y

    print(z)
    # z = z.numpy() # numpy ONLY CPU
    # so:

    # z = z.to("cpu")
    # z = z.numpy()

    z = z.cpu().numpy()
    print(type(z))

x = torch.ones(5 , requires_grad=True) # defualt is False
# needed for optmization so during trainnig it have to calacalte gradaide to this tensor 