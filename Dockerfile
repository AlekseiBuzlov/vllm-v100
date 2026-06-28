FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# On V100 we cannot rely on BF16, so dtype is set to "half" (fp16) in compose.
ENTRYPOINT ["vllm", "serve"]
