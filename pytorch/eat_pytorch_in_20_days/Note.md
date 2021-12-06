# PyTorch笔记

### 张量和numpy数组

可以用numpy方法从Tensor得到numpy数组，也可以用torch.from_numpy从numpy数组得到Tensor。

<font color=red size=3>这两种方法关联的Tensor和numpy数组是共享数据内存的。</font>

如果改变其中一个，另外一个的值也会发生改变。

如果有需要，可以用张量的**clone方法**拷贝张量，中断这种关联。