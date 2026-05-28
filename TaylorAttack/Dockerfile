FROM lmsysorg/sglang:v0.5.12

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/workspace/TaylorMLP/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/TaylorMLP/.cache/huggingface

WORKDIR /workspace/TaylorMLP

COPY TaylorAttack/requirements.txt /tmp/taylorattack-requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/taylorattack-requirements.txt

CMD ["/bin/bash"]
