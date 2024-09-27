# 梯度反向传播

```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# calculate a scalar loss related to the parameters of the model
loss = ...

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

首先，定义优化器 `optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)` 该优化器会根据各个可训练参数张量`（model.parameters()）`对应的梯度，来更新这些可训练参数张量的值，该优化过程通过调用 `optimizer.step()` 实现。

定义完成优化器后，根据具体任务定义损失函数。损失函数必须是各个可训练参数张量的函数，这样，进行自动梯度计算时，才能正确地将梯度信息反向传播到可训练参数张量。

计算得到损失函数后，在进行自动梯度计算之前，必须利用 `optimizer.zero_grad()` 先将各个张量现有的梯度全部置零。这是因为 Pytorch 在每次 `backward()` 之后，都会将新的梯度信息累加到现有的梯度信息上，而不是覆盖掉现有的梯度信息。

然后就是以损失函数为输出量，各个训练参数张量为输入量，计算梯度。该步对应 `loss.backward()`。需要注意的是，进行 backward 后，如果没有设置额外的 `retain_graph` 参数，计算图就会被销毁，此时就不能进行第二次 backward 了。也正如上一段中提到的，如果设置了 `retain_graph=True` 且进行了第二次 backward，则梯度会变成只进行一次 backward 的两倍。

最后让优化器利用梯度对各个张量的值进行更新： `optmizer.step()`。

## pytorch 求导相关

torch.tensor 具有如下属性：

* 查看 是否可以求导 `requires_grad`
* 查看 运算名称 `grad_fn`
* 查看 是否为叶子节点 `is_leaf`
* 查看 导数值 `grad `

PyTorch提供两种求梯度的方法：

* `backward()` 给叶子节点填充.grad字段
* `torch.autograd.grad()`直接返回梯度

使用`backward()`函数反向传播计算tensor的梯度时，并不计算所有tensor的梯度，而是只计算满足这几个条件的tensor的梯度：

1. 类型为叶子节点
2. `requires_grad=True`
3. 依赖该tensor的所有tensor的`requires_grad=True`

所有满足条件的变量梯度会自动保存到对应的grad属性里。

```python
x = torch.tensor(2., requires_grad=True)

a = torch.add(x, 1)
b = torch.add(x, 2)
y = torch.mul(a, b)

# 使用backward求导
y.backward()
print(x.grad)
>>>tensor(7.)

print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print("grad: ", x.grad, a.grad, b.grad, y.grad)

>>>requires_grad:  True True True True
>>>is_leaf:  True False False False
>>>grad:  tensor(7.) None None None

#使用autograd.grad()
grad = torch.autograd.grad(outputs=y, inputs=x)
print(grad[0])
>>>tensor(7.)
```

对于Jacobian矩阵的求导

![img.png](img.png)