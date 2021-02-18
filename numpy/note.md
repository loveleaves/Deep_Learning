# some notes about numpy  

# 索引  
### 注意  

NumPy切片创建视图而不是复制，就像内置Python序列（如string，tuple和list）一样。  
从大数组中提取一小部分时必须小心，这在提取后变得无用，因为提取的小部分包含对大原始数组的引用，  
其内存将不会被释放，直到从其派生的所有数组被垃圾收集。在这种情况下，copy()建议使用明确的。  

#### 代码  
> x,y=np.mgrid[1:3:1,2:4:1]  
> grid=np.c_[x.ravel(),y.ravel()]  
> print(grid)  
> print(grid[:,0])  

#### 结果  
> [[1 2]  
> [1 3]  
> [2 2]  
> [2 3]]  
> [1 1 2 2]  
