# 实验5：LLM Agent开发

## 实验目标与要求

利用LangGraph编程实现一个笔记管理员Agent，其将会自动搜索用户提供的文献，生成一段笔记，存入到飞书笔记库中。

**要求**：

1. **工具调用与自动路由：用户提供关键词、标题或链接后，Agent需要搜索和读取链接内容，并且判断用户提供的内容是否需要记入笔记。**如用户输入“ResNet的主要机制是什么”，LLM Agent通过调用搜索工具和读取工具，判断为需要记入笔记；但如果用户输入“什么乐器容易入门”这样与笔记本主题无关的内容，则应该只回答用户，而不需要记入笔记。

2. **提示词工程与工作流：如果需要记入笔记，Agent将生成一段固定格式的摘要，并插入到飞书笔记中的合适位置。**根据上一步的搜索结果，LLM Agent需要总结得到一段简洁易懂的摘要（还包括文章标题、链接等），作为对用户的回答。然后需要读取飞书笔记中已有的小标题，LLM Agent决定这篇摘要属于哪个小标题，最后调用飞书API进行插入。

3. **加分项：多模态能力：**链接中或搜索结果中有相关图片，Agent可以选择其中合适的一张图片插入到笔记中。



**例如：**

用户输入“pi0.6这篇文章的核心观点是什么”，Agent通过搜索网络信息后，决定需要将内容记入笔记，并且生成一个如下所示的结构良好的笔记摘要。然后调用飞书API读取笔记中的子标题，Agent决定把新摘要插入到“VLA-RL”这个小标题之下，最后调用飞书API完成插入。

![截屏2025-12-08 21.57.17](pictures/截屏2025-12-08 21.57.17.png)

## 实验准备与提交

**对于飞书文档的准备：**

首先在飞书中创建一个云文档，写入你的笔记标题和至少三个小标题，如一级大标题为《大模型学习笔记》，二级小标题为 “计算机视觉”，“大语言模型”，“多模态大模型”等。

参照飞书的[开发文档](https://open.feishu.cn/document/server-docs/api-call-guide/calling-process/overview)，同时需要获取你笔记文档的[访问凭证](https://open.feishu.cn/document/server-docs/api-call-guide/calling-process/get-access-token)。即可让程序读取你的笔记内容与修改内容。



在[硅基流动](https://docs.siliconflow.cn/cn/userguide/quickstart)（按照链接中的指南上手）或其他LLM API提供商上注册账号，选择一款模型，如DeepSeek-V3.2作为实验用的LLM。

为了保证自己的API key不泄露，需要在本地的`.env`文件中写入你的API key，如：

```
OPENAI_API_KEY="YOUR_API_KEY_FROM_CLOUD_SILICONFLOW_CN"
OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

在python中可以通过dotenv加载.env中的环境变量

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI
client = openai.OpenAI(
  api_key=os.environ["OPENAI_API_KEY"], 
  base_url=os.environ["OPENAI_BASE_URL"]
)
```

然后在本地安装LangChain与LangGraph等必备的python库，即可开始你的编程。

编程与实验结束后，请上交你的代码和一个PDF文件展示你的成果（包括流程图讲解）。