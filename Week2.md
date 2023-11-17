# Week2

## def 函数

- 固定参数无赋值，默认参数有赋值，可选参数元组有星号（*）
- 默认参数在非默认参数之后
- 可选参数必须在参数列表最后一个
- 函数如果不调用，则函数体中的代码都不会执行，伪代码，让一个函数指代多行代码
  - 执行对应函数：`函数名()`
- 什么时候用函数？有大量重复，代码太长时。函数式编程
- 一行一行的堆砌-->面向过程编程
- 参数：
  - 参数名第一次出现在def 函数名后的括号里，也叫变量，可以赋不同的值
  - 在函数体中其对应位置type in参数名
  - 执行函数时传入具体的值，有几个参数就要传几个值，顺序重要（按位置传参）；按关键字传参`函数名(参数1=值1，参数2=值2...)` 这时顺序没有那么重要；二者mixed也可以，但是要确保位置传在前 关键字传参在后
  - 未传入值前--形式参数；传入值后--实际参数
- 使用`open('文件路径')`函数可以打开文件
- 动态参数：
  - 什么时候用？定义的参数不知道有多少个时
  - `*args`:适用于*位置*传参数；args相当于元组，当不传参-->空
  - `** kwargs`：适用于*关键字*传参数；字典类型接收参数，不传参时-->空字典
  - 一般在开发过程中都用上比较合适
  - 顺序：非默认参数，默认参数，*args，**kwargs
- .format()
- 函数返回值：希望得到函数结果时，使用
  - 如果没有返回值/只写了`return`/`return None`，返回为None
  - 返回值可以是任意类型：列表、字典、元组、变量、int
  - 如果return后有逗号，则默认返回的类型为*元组*
  - 函数一旦遇到return对退出函数，中止执行，不执行函数体中return后的代码
    - 和break不同，break只是终止执行当前函数，但是同级的其他函数还可以运行

```python
def 函数名(参数名):
  ‘’‘
  函数注释
  ’‘’
  listName=[]
  for i in items:
    计算
    listName.append(...)
  return listName
result=函数名(参数赋值) #执行函数，接收返回值
print(result)
```

- 参数的内存地址：
  - 使用python中`id(参数名)`来查看
  - 函数执行时传参-->传递的是内存地址
    - 好处：节省内存，便利了函数内部可变参数(list, dict, set)的元素内部的增删改【在其他的编程语言中可能会不同】
  - 函数返回（return的内容）的也是内存地址
    - 引用计数器：count有几个变量指向这块内存地址
      - 函数*执行完*后，所有<u>函数内部的变量</u>需要被删掉，引用计数-1
      - python内部有*缓存和驻留机制*，如果连续给不同变量赋相同的值，指向的内存地址是一样的
- 函数的默认值：
  - 执行函数时，给默认参数传递新值，则该参数print出后是新的值而非括号里的默认值
  - 如果括号里有默认参数，会存在一个区域储存默认参数的值
- 动态参数：
  - 执行函数是`*` 和`**`也可以**对应**使用
  - 形参、实参都使用`*` 和`**`传参时，数据会重新拷贝一份
- 函数可以被看作变量、元素(可以被放在由多个元素组成的容器[字典、list等]中)
  - application：

```python
function_dict={#注意这里是函数名，不执行函数
  1: [function_name1, [参数值1,参数值2,...]],
  2: [function_name2, [参数值1,参数值2,...]],
  3: [function_name3, [参数值1,参数值2,...]]
}
choice=input("输入序号")
fun=function_dict.get(choice) 
fun()
# 把需要后面上传的参数放在对应的list中，就不用担心执行函数的时候会因为参数个数问题报错
#################################################################
function_list=[
  function_name1, 
  function_name2, 
  function_name3]
for item in function_list:
  item() #执行列表中的函数
# 前提：list/dict中函数的参数需要相同
```

- return和print的区别

## LLM Retrieval相关

### Evaluation

- MRR：<https://lakshyakhandelwal.substack.com/p/mean-reciprocal-rank?r=d7ga5&utm_campaign=post&utm_medium=web>
  - 用于评估搜索引擎/QA系统/推荐系统的效率
- Hit Rate

### Prompt

- prompt的设计与优化：
  - 需要包括上下文中相关的关键词进行提示，或者 在prompt中提供足够的上下文
  - 问题更加具有针对性
  - 要求格式化输出：使用列表、表格等
  - 限制输入/输出长度，使得输入长度在模型可接受范围
    - 使用特定参数 `max_tokens(...)`
  - 通过设置关键字：准确地、简洁地、正确地 使模型生成结果优化
  - 设置temperature, 其值越小，生成的可靠度越高
  - 复杂任务可以将其分解为多个简单的任务以引导生成的结果，逐步精细化
  - 为模型提供示例，帮助理解输出的预期
  - 为模型生成的内容提供反馈和评价
  - 提前预知会提供的错误信息，在prompt中明确要避免
  - 在prompt中提供需要解决任务的背景，有利于提高模型泛化能力
- example：

```python
prompt=(
  "相关的上下文信息如下：\n"
  "-------------------"
  "{context_str}\n"
  "-------------------"
  "请使用给定的上下文信息，不要使用先验知识，回答下列问题。\n"
  "问题：{query_str}\n"
  "回答: "
)
```

## 环境配置

### CUDA

- 没有下CUDA怎么办？下载pytorch2.0及以上版本，里面含有cuda。
- 设置环境, 多卡运行时：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES']='2, 3,...'
```

- 在终端中查看python进程的PID，并kill掉释放空间
  
```shell
nvidia-smi #会显示对应的PID
sudo kill -9 PIDNumber
```

- 释放缓存

```python
import gc
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
report_gpu()
```