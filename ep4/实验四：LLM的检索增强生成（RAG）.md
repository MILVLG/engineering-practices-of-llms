# 实验四：LLM的检索增强生成（RAG）

## 实验目标与要求

利用[LangChain](https://docs.langchain.com/oss/python/langchain/install)编程实现LLM的检索增强生成（RAG），使其能够回答用户针对[《2025年人工智能指数报告》](https://hai.stanford.edu/assets/files/hai_ai_index_report_2025_chinese_version_061325.pdf)这个文档的提问。

**要求：**

1.自己设计各种各样的用户问题，检查RAG系统的回答与真实答案的区别，探索RAG能力的边界

2.探索RAG的最优配置（调整chunk size，retriever等）

3.使用AdvancedRAG、IterativeRAG、GraphRAG等解决基础RAG无法解决的问题

4.加分项：构建与用户对话的长时记忆，比如能够回答“我们之前聊了什么人工智能的应用领域？”

## 实验准备与提交

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

然后在本地安装LangChain等必备的python库，即可开始你的编程。

编程与实验结束后，请上交你的代码和一个PDF文件展示你的成果。

