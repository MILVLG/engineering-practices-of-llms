# 安装包、数据等已提前存至/data下

## 第一步: Conda安装

进入docker并安装 Conda
(i取0-7)
```bash
docker exec -it cann-i /bin/bash
cd /data
bash Miniconda3-latest-Linux-aarch64.sh
```
重启终端

# 第二步: 环境配置

```bash
docker exec -it cann-i /bin/bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n llm_env python=3.10 -y
```

启动环境

```bash
conda activate llm_env
cd /home
cp -r /data/engineering-practices-of-llms-main /home/test_student/
cd test_student/engineering-practices-of-llms-main/
pip install -r requirements.txt 
```


# 第三步: 启动jupyter notebook 以及 npu监控

port请随意选择，例如8888。不冲突即可。
```bash
jupyter notebook --allow-root --port=
```

另开一个终端，用于观察npu情况
```bash
docker exec -it cann-i /bin/bash
watch -n 0.1 -d npu-smi info
```

# 第四步: 移动所需数据集

使用以下命令将准备好的数据移动到工作路径下

```bash
cp -r /data/engineering-practices-of-llms/experiment0/data /home/test_student/engineering-practices-of-llms-main/experiment0
```
