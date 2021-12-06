# 对比常用张量复制操作

[TOC]

## 1. clone

返回一个和源张量同`shape`、`dtype`和`device`的张量，与源张量**不共享数据内存**，但提供**梯度的回溯**。

（1）定义

``` python3
import torch

a = torch.tensor(1.0, requires_grad=True, device="cuda", dtype=torch.float64)
a_ = a.clone()
print(a_)   # tensor(1., device='cuda:0', dtype=torch.float64, grad_fn=<CloneBackward>)
```

**注意**：`grad_fn=<CloneBackward>`，说明`clone`后的返回值是个中间variable，因此支持梯度的回溯。因此，`clone`操作在一定程度上可以视为是一个identity-mapping函数。

（2）梯度的回溯

`clone`作为一个中间variable，会将梯度传给源张量进行叠加。

``` python3
import torch

a = torch.tensor(1.0, requires_grad=True)
y = a ** 2 
a_ = a.clone()
z = a_ * 3
y.backward()
print(a.grad)   # 2
z.backward()
print(a_.grad)　　　# None. 中间variable，无grad
print(a.grad)    #　5. a_的梯度会传递回给a，因此2+3=5
```

但若源张量的`require_grad=False`，而`clone`后的张量`require_grad=True`，显然此时不存在张量回溯现象，`clone`后的张量可以求导。

``` python3
import torch

a = torch.tensor(1.0)
a_ = a.clone()
a_.requires_grad_()

y = a_ ** 2
y.backward()
print(a.grad)   # None
print(a_.grad)   # 2.  可得到导数
```

（3）张量数据非共享

``` python3
import torch

a = torch.tensor(1.0, requires_grad=True)
a_ = a.clone()

a.data *= 3
a_ += 1

print(a)   # tensor(3., requires_grad=True)
print(a_)  # tensor(2., grad_fn=<AddBackward0>).  注意grad_fn的变化
```

综上论述，`clone`操作在不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下。

## 2. detach

`detach`的机制则与`clone`完全不同，即返回一个和源张量同`shape`、`dtype`和`device`的张量，与源张量**共享数据内存**，但不提供**梯度计算**，即`requires_grad=False`，因此脱离计算图。

（1）定义

``` python3
import torch

a = torch.tensor(1.0, requires_grad=True, device="cuda", dtype=torch.float64)

a_ = a.detach()
print(a_)   # tensor(1., device='cuda:0', dtype=torch.float64)
```

（2）脱离原计算图

``` python3
import torch

a = torch.tensor(1.0, requires_grad=True)
y = a ** 2 
a_ = a.detach()
print(a_.grad)    # None，requires_grad=False
a_.requires_grad_()  # 强制其requires_grad=True，从而支持求导

z = a_ * 3
y.backward()
z.backward()

print(a.grad)    #　2，与a_无关系
print(a_.grad)   #
```

可见，`detach`后的张量，即使重新定义`requires_grad=True`，也与源张量的梯度没有关系。

（3）共享张量数据内存

``` python3
import torch

a = torch.tensor(1.0, requires_grad=True)
a_ = a.detach()

print(a)    # tensor(1., requires_grad=True)
print(a_)   # tensor(1.)

a_ += 1   
print(a)     # tensor(2., requires_grad=True)
print(a_)    # tensor(2.)

a.data *= 2
print(a)    # tensor(4., requires_grad=True)
print(a_)    # tensor(4.)
```

综上论述，detach操作在共享数据内存的脱离计算图，所以常用在神经网络中仅要利用张量数值，而不需要追踪导数的场景下。

## 3. clone和detach联合使用

clone提供了非数据共享的梯度追溯功能，而detach又“舍弃”了梯度功能，因此clone和detach意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。

置于是先clone还是先detach，其返回值一样，一般采用tensor.clone().detach()。

## 4. new_tensor

new_tensor可以将源张量中的数据复制到目标张量（数据不共享），同时提供了更细致的device、dtype和requires_grad属性控制：

``` python3
new_tensor(data, dtype=None, device=None, requires_grad=False) 
```

**注意**：其默认参数下的操作等同于`.clone().detach()`，而`requires_grad=True`时的效果相当于`.clone().detach()requires_grad_(True)`。上面两种情况都推荐使用后者。

## 5. copy_

`copy_`同样将源张量中的数据复制到目标张量（数据不共享），其`device`、`dtype`和`requires_grad`一般都保留目标张量的设定，仅仅进行数据复制，同时其支持broadcast操作。

``` python3
a = torch.tensor([[1,2,3], [4,5,6]], device="cuda")
b = torch.tensor([7.0,8.0,9.0], requires_grad=True)
a.copy_(b)
print(a)   # tensor([[7, 8, 9], [7, 8, 9]], device='cuda:0')  
```

