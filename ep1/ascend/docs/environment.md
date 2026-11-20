# 第一步: 安装 MindSpeed

新建虚拟环境并安装 MindSpeed

```bash
conda create -n llm_pretrain python=3.10 -y
conda activate llm_pretrain

git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 4ea42a23
pip install . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

# 第二步: 安装 MindSpeed-LLM

本实验以`./MindSpeed-LLM/`为工作根目录，请先进入该目录并安装 MindSpeed-LLM

```bash
cd ./MindSpeed-LLM
pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

# 第三步: 安装 apex

```bash
cd ..
git clone https://gitee.com/ascend/apex.git
cd apex
bash scripts/build.sh --python=3.10
pip install --force-reinstall apex/dist/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl
cd ../MindSpeed-LLM
```

# 第四步: 下载预训练数据集

使用以下命令下载预训练数据集到`/data/datasets/`目录下

```bash
obsutil cp obs://hangdian/Ultra-FineWeb-ShaoZW_subset /data/datasets/ -r -f
```
