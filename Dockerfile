# NVIDIA's ARM64/Blackwell PyTorch build is retained, not replaced by PyPI torch.
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
FROM ${PYTORCH_IMAGE}

WORKDIR /app
ENV HF_HOME=/cache/huggingface \
    PATH="/opt/laya-warehouse-venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

RUN python -m venv --system-site-packages /opt/laya-warehouse-venv \
    && python -m pip install --no-cache-dir \
        "huggingface_hub>=0.20" \
        "numpy>=1.20" \
        "safetensors>=0.4" \
        "transformers>=4.45" \
    && python -m pip install --no-cache-dir --no-deps "laya==0.3.3"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY benchmarks/compositional-coverage/manifests ./benchmarks/compositional-coverage/manifests
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -c "import laya, torch; print('Laya:', laya.__version__, 'NVIDIA torch:', torch.__version__)"

ENTRYPOINT ["laya-warehouse"]
CMD ["--help"]
