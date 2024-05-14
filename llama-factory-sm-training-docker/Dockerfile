# 根据https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html 找到合适的基础镜像
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# 允许使用所有显卡
ENV NVIDIA_VISIBLE_DEVICES="all"

# 下载并安装LlamaFactory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git --branch v0.7.0 --single-branch
WORKDIR /workspace/LLaMA-Factory
RUN pip install -e .[metrics,deepspeed,bitsandbytes]

# 安装SageMaker-Training-Toolkit的pytorch版本
RUN pip install sagemaker-pytorch-training

COPY llama3-sft.sh .
COPY ad-hoc-inference.py . 