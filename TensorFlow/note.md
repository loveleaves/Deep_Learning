# Note



### keras使用中间网络层

``` python
middle = Model(inputs=model.input,outputs=model.get_layer('cov1').output)
# inputs=输入层，outputs=输出层，model.get_layer(name)获取指定层，网络可设置name参数
result = middle.predict(x_test)[0]
result = middle.evaluate(x_test)
```

