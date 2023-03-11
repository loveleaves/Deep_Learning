# Notes

## 高速源地址：  

> 阿里云 速度最快 http://mirrors.aliyun.com/pypi/simple/   
> 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/  
> 豆瓣(douban) http://pypi.douban.com/simple/  
> Python官方 https://pypi.python.org/simple/  
> v2ex http://pypi.v2ex.com/simple/  
> 中国科学院 http://pypi.mirrors.opencas.cn/simple/  
> 清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/  



## 环境配置

**安装GPU版本(conda虚拟环境下)**
注意cudatoolkit和cudnn版本对应
conda search cudatoolkit 和conda search cudnn查询可使用版本
conda install cudatoolkit==10.0 cudnn==7.6.5（tf1.14可用）

**如安装TF框架 tensorflow-CPU**  

> pip install -U tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple  

**在jupyter中添加新的环境  **

如在conda中创建虚拟环境TF  

> conda -n tf2 python=版本号  
> conda activate tf2  
> conda install nb_conda  
> conda install ipykernel  
> python -m ipykernel install --user --name tf2 --display-name "tf2"
> conda remove -n tf2 --all
> conda env export > enviroment.yaml
> conda create -f enviroment.yaml


**查看 Jupyter notebook kernel  **

> jupyter kernelspec list  

**删除 jupyter 内核  **

> jupyter kernelspec remove kernelname  



## DL相关

[数据预处理方法](./data_preproccess.md)

[Python笔记](./python.md)
