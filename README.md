# vLLM + Qwen2.5-Coder-32B на 2× Tesla V100

Запуск [Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8)
через vLLM (OpenAI-compatible API) и [Open WebUI](https://github.com/open-webui/open-webui)
на двух GPU NVIDIA Tesla V100 (32 GB).

## Установка Docker и Docker Compose

Инструкции ниже — для Ubuntu / Debian в CLI. На сервере с V100 обычно стоит Ubuntu 22.04 или 24.04.
Для других дистрибутивов см. [официальную документацию Docker](https://docs.docker.com/engine/install/).

### Проверка, установлен ли Docker

```bash
docker --version
docker compose version
```

Если обе команды отрабатывают — этот раздел можно пропустить и перейти к [запуску](#запуск).

### Docker Engine + Compose plugin (Ubuntu / Debian)

Современный `docker compose` ставится как плагин вместе с Docker Engine — отдельный бинарник `docker-compose` не нужен.

```bash
# Удалить старые пакеты, если были
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Зависимости
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# GPG-ключ Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Репозиторий Docker
# Для Debian замените "ubuntu" на "debian" в URL ниже
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Запуск Docker без `sudo` (нужен повторный вход в SSH-сессию после команды):

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Проверка:

```bash
docker run --rm hello-world
docker compose version
```

### NVIDIA Container Toolkit (обязательно для GPU)

Без него контейнеры не увидят видеокарты. Подробнее:
[NVIDIA Container Toolkit — install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка проброса GPU в контейнер:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Должен вывестись список GPU (в т.ч. две V100).

## Почему GPTQ Int8 на V100?

V100 — это Volta (compute capability 7.0). vLLM поддерживает **GPTQ** на Volta, но **не**
поддерживает AWQ / FP8 / Marlin kernels. Поэтому GPTQ Int8 здесь — правильный выбор; AWQ на V100 лучше не использовать.

- `--dtype half` (fp16) обязателен: у V100 нет быстрого пути для BF16.
- `--tensor-parallel-size 2` практически обязателен: Int8-модель весит ~33 GB, а одна
  V100 имеет 32 GB, плюс нужны KV-cache и служебная память.
- Ожидайте *рабочий*, но не «современно быстрый» инференс: быстрый Marlin kernel для GPTQ/AWQ/FP8
  на Volta не поддерживается, поэтому пропускная способность заметно ниже, чем на A100/4090/H100.

## Файлы

```
.
├── Dockerfile          # vllm/vllm-openai:latest с env для HF cache
├── docker-compose.yml  # qwen-vllm (порт 8000) + open-webui (порт 3000)
├── .env                # HF_TOKEN + MODEL_ID (в .gitignore)
└── .env.example        # шаблон
```

## Запуск

1. Скопируйте шаблон env и отредактируйте его:

```bash
cp .env.example .env
# затем укажите HF token в .env (для публичных моделей необязательно, но помогает избежать rate limit)
```

2. Убедитесь, что Docker видит GPU (см. [проверку выше](#nvidia-container-toolkit-обязательно-для-gpu)):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

3. Соберите и запустите:

```bash
docker compose up -d --build
```

4. Следите за загрузкой модели (первый запуск скачает ~33 GB):

```bash
docker logs -f qwen-vllm
```

## Доступ

- Open WebUI: http://localhost:3000
- vLLM API (напрямую):

```bash
curl http://localhost:8000/v1/models
```

- Тестовый chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "qwen2.5-coder-32b-int8",
    "messages": [
      { "role": "user", "content": "Write a NestJS service with Redis cache for user profiles." }
    ],
    "temperature": 0.2,
    "max_tokens": 1024
  }'
```

## Настройка

- `--max-model-len 16384` — безопасный старт. Если модель загрузилась и `nvidia-smi` показывает свободную память,
  можно попробовать `32768`. Контекст 131k для 32B-модели на 2× V100 32 GB почти наверняка
  не влезет из-за KV-cache.
- Для кодинга в UI начинайте с `max_tokens` 1024–2048, контекст 16k, temperature 0.1–0.3.

## Диагностика

### Падает на quantization

Пусть vLLM возьмёт quantization из `quantization_config` модели — уберите явный флаг
в `docker-compose.yml`:

```yaml
      - --dtype
      - half
      - --tensor-parallel-size
      - "2"
      - --max-model-len
      - "16384"
```

(то есть удалите строки `--quantization` / `gptq`).

### Ошибки NCCL / NVLink

Проверьте топологию — между GPU должны быть связи `NV1` / `NV2`:

```bash
nvidia-smi topo -m
```

Временные диагностические переменные окружения для сервиса `qwen-vllm`:

```yaml
environment:
  NCCL_DEBUG: "INFO"
  NCCL_P2P_DISABLE: "0"
  NCCL_IB_DISABLE: "1"
```

Если всё равно не стартует, можно отключить P2P (будет медленнее):

```yaml
  NCCL_P2P_DISABLE: "1"
```

### Образ `latest` не стартует (CUDA/драйвер)

`vllm/vllm-openai:latest` может требовать свежий NVIDIA driver. При ошибке
`CUDA driver version is insufficient` обновите драйвер на хосте или зафиксируйте более старый
образ vLLM. Быстрая проверка:

```bash
nvidia-smi
docker run --rm --gpus all vllm/vllm-openai:latest nvidia-smi
```

## Остановка

```bash
docker compose down          # volumes сохраняются (кэш модели + данные UI)
docker compose down -v       # также удалить volumes
```
