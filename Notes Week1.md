## LLaMA Index

- structure data with indexes, connect external data sources
- RAG
- queries act based on the index
- stages: loading->indexing->storing->querying->valuation
- Node: a chunk of a source of document
- connector(reader): ingested data from different sources and format into doc and nodes
- index: easy to retrieve as *vector embedding*(numerical representations) after filtering to get the relevant data stored in a vector store
- retrievers: 有效率的根据query和index检索
- node postprocessors: 得到retrieved nodes并transformations, filtering, re-ranking
- chat-engines:多轮对话
- query engines：端对端pipeline, queries+response+reference content-->LLM

```python
from llama_index.node_parser import SimpleNodeParser

# load documents

# parse nodes
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# build index
index = VectorStoreIndex(nodes)
```

- `text_splitter`: split text into chunks,it has sentence splitter, token text splitter, code splitter

## Shell Command

- vim文档编辑和退出方式(一般使用 `:wq`保存并退出)
- SSH key generation

## 异步方法

- await:等到等待对象返回结果，即可执行后面的代码
- async:在单个线程实现多任务切换，提高并发性

```python
import asyncio
async def function1():
  return ("协程1")
async def function2():
  return ("协程2")
task=[function1(),function2()]
async.run(task)
```

- 模拟事件循环

```python
loop=events.get_running_loop()
loop.run_until_complete(asyncio.wait(task)) #添加任务,直到传入的可等待对象完成为止
future=loop.create_future() #代表异步完成的最终结果，一个特殊的等待对象
loop.close() #关闭循环
```

- asyncio.sleep(2)：
  - 使协程暂停一段时间后，继续工作
  - 用于模拟耗时操作 或 调整协程运行的时间间隔,合理分配要竞争的任务
  - 可用来创建定时事件
- reference material：<https://blog.csdn.net/weixin_43665662/article/details/130218312>

## FastAPI

- in JSON format
- What is API:
  - 应用程序编程接口,两个应用程序之间的服务合约定义request和response
  - 由客户端和服务端组成
  - 4中工作方式
    - soap api:使用XML交换信息
    - rpc api
    - websocket api：JSON
    - RESTapi：无状态
- Install:

```shell
pip3 install fastapi
pip3 install uvicorn[standard] #use to run our web server
#if we want to use web server
uvicorn fileName:objectInstanceName #without file type postfix
```
Then we can find the UI on <http://127.0.0.1:8000>

- define endpoint, a communication channel with URL by HTTP
  - the primary HTTP request types:
    - GET: retrieve data from server, read-only, server state doesn't change
    - POST: send data to the server for creating new resources
    - PUT: update an existing resource, OR create a new one if hasn't exist
    - DELETE: delete a resource by URL
  - in python, we need decorator to define the endpoints
- Path Parameter
- Query Parameter
- Request Body
