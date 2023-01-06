import torch

## create an one dimentional tensor
x = torch.arange(12)
print(x)

## get the shape of data
print(x.shape)

## get the number of elements
print(x.numel())

## reshape the tensor
x = x.reshape(2,6) # attention: reassigne the value to the variation
x = x.reshape(6,-1) # use -1, the tensor can calculate the shape itself
print("reshaped",x)

## torch.zeros(tensor.Size()), torch.ones(tensor.Size()). Set all to one or zero
print(torch.zeros((2,2,3)))
print(torch.ones((2,3,4)))

## Each element is randomly sampled from a standard Gaussian distribution (normal distribution)
## with a mean of 0 and a standard deviation of 1.
print(torch.randn((2,3,4)))

## define a tensor manually
print(torch.tensor([[1,2,3],[2,3,4]]))


## concatenate two tensors
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

X = torch.arange(12).reshape(2,2,3)
Y = torch.arange(12,24).reshape(2,2,3)
print(torch.cat((X,Y), dim=0))

## sum all elements
print(X.sum())

## broadcasting mechanism for all operations, inkl. ==, >, <
X = torch.arange(12).reshape(2,2,3)
Y = torch.arange(3)
print(X+Y)
'''
X:
[[0,1,2],
 [3,4,5]
 
 [6,7,8],
 [9,10,11]]
 
 Y:
 [[0,1,2]]

Process:
Y-->
[[0,1,2],
 [0,1,2]
 
 [0,1,2]
 [0,1,2]]

X+Y=
tensor([[[ 0,  2,  4],
         [ 3,  5,  7]],

        [[ 6,  8, 10],
         [ 9, 11, 13]]])
'''
## try operations between nonsingelton in certain dimension 
x = torch.tensor([1,2,3,4])
y = torch.tensor([[2,3,4],[2,3,3]])
print(x+y)
# error: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1.

## save the storage while caculating the X+Y
## use += or Z[:]
#1.
X += Y
#2.
Z = torch.zeros_like(Y)
Z[:] = X+Y
# compare id before and after the caculation id(Y)

## Convert to other Python objects
# numpy.ndarray <--> torch.tensor
A = X.numpy()
B = torch.tensor(A)
# torch.tensor <--> float, int in python
a = torch.tensor([3.5])
a.item(), float(a), int(a)