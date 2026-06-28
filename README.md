# llama.cpp + Qwen2.5-Coder-32B на 2× Tesla V100 (SM70)

OpenAI-совместимый инференс [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
в формате **GGUF** через **llama.cpp (`llama-server`)** с CUDA-бэкендом, собранным
специально под **NVIDIA Tesla V100 SXM2 32 GB (Volta, compute capability 7.0 / sm_70)**.
Плюс [Open WebUI](https://github.com/open-webui/open-webui) для удобного UI.

API совместим с Continue, Cline, Aider и Open WebUI:

```
POST http://<host>:8000/v1/chat/completions
```

---

## Почему не vLLM (объяснение проблемы)

Текущий деплой на vLLM падал, потому что современные сборки PyTorch/vLLM **больше не
содержат kernels для Volta (sm_70)**. Бинарь PyTorch скомпилирован только под:

```
sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
```

V100 — это `sm_70`, поэтому нужных kernels просто нет. Отсюда и вся цепочка ошибок:

```
NCCL internal error
Failed to find reverse path
WorkerProc failed to start
Engine core initialization failed
```

Это **следствия**, а не причина: воркеры не стартуют, потому что под GPU нет кода.

### Почему llama.cpp (Вариант A, выбранный)

- **Нет зависимости от PyTorch.** llama.cpp компилирует собственные CUDA-kernels.
- Мы фиксируем `CMAKE_CUDA_ARCHITECTURES=70`, то есть бинарь содержит **реальные sm_70 kernels** для V100.
- **Не использует NCCL.** Мульти-GPU работает через собственный CUDA peer-to-peer (по NVLink). Тех ошибок NCCL больше не будет.
- Отличная и стабильная поддержка Volta, простой деплой.
- Один минус: на V100 это «рабочий», а не «топовый» по скорости инференс (нет fast-kernels уровня A100/H100), но для кодинга это вполне usable.

> Вариант B (остаться на vLLM) описан в конце файла — он сложнее и хрупче, поэтому не рекомендуется.

---

## Диагностика текущего окружения

Прежде чем менять стек, полезно зафиксировать, что было (и проверить драйвер хоста).

На хосте:

```bash
nvidia-smi                       # модель GPU + версия драйвера + CUDA driver API
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi topo -m               # топология: между двумя V100 должны быть NV# (NVLink)
```

Если у вас ещё запущен старый vLLM-контейнер — версии внутри него:

```bash
docker ps
docker exec <vllm_container> python -c "import torch, vllm; \
  print('torch', torch.__version__); \
  print('vllm', vllm.__version__); \
  print('cuda', torch.version.cuda); \
  print('arch_list', torch.cuda.get_arch_list())"   # тут НЕ будет sm_70
docker exec <vllm_container> nvidia-smi
```

`torch.cuda.get_arch_list()` без `sm_70` — это и есть подтверждение корневой причины.

---

## Требования к хосту

1. **NVIDIA-драйвер** для CUDA 12.1 (используется в образе) — `>= 525.60.13`.
   Проверка: `nvidia-smi` (поле *Driver Version*).
   Если драйвер старее — в `Dockerfile` поменяйте оба базовых образа с `12.1.0` на `11.8.0`.
2. **Docker** + **Docker Compose plugin** (см. установку ниже).
3. **NVIDIA Container Toolkit** (проброс GPU в контейнеры).

---

## Установка Docker и Docker Compose

Инструкции для Ubuntu / Debian в CLI. Для других дистрибутивов см.
[официальную документацию Docker](https://docs.docker.com/engine/install/).

### Проверка, установлен ли Docker

```bash
docker --version
docker compose version
```

Если обе команды отрабатывают — переходите к [NVIDIA Container Toolkit](#nvidia-container-toolkit-обязательно-для-gpu).

### Docker Engine + Compose plugin (Ubuntu / Debian)

```bash
# Удалить старые пакеты, если были
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Для Debian замените "ubuntu" на "debian" в URL ниже
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Запуск без `sudo` (нужен повторный вход в SSH-сессию):

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

### NVIDIA Container Toolkit (обязательно для GPU)

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

Проверка проброса обеих V100 в контейнер:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

В выводе должны быть **две** Tesla V100.

---

## Файлы

```
.
├── Dockerfile          # сборка llama.cpp из исходников, CUDA-бэкенд, arch sm_70
├── docker-compose.yml  # llama-server (порт 8000) + open-webui (порт 3000)
├── .env                # модель/квант/токен (в .gitignore)
└── .env.example        # шаблон
```

---

## Запуск

1. Подготовьте `.env`:

```bash
cp .env.example .env
# при желании укажите HF_TOKEN и/или поменяйте MODEL_QUANT (Q5_K_M / Q4_K_M)
```

2. Соберите образ (первая сборка llama.cpp из исходников занимает несколько минут)
   и запустите:

```bash
docker compose up -d --build
```

3. Следите за загрузкой модели (первый запуск качает ~23 GB для Q5_K_M):

```bash
docker logs -f llama-server
```

Дождитесь строк вида:

```
load_tensors: offloaded 65/65 layers to GPU
...
main: server is listening on http://0.0.0.0:8000
```

`offloaded 65/65 layers to GPU` означает, что CPU-offload нет — всё на GPU.

---

## Доступ

- Open WebUI: http://localhost:3000
- API llama.cpp (OpenAI-совместимый):

```bash
curl http://localhost:8000/v1/models
```

Тестовый запрос:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "qwen2.5-coder-32b",
    "messages": [
      { "role": "user", "content": "Write a NestJS service with Redis cache for user profiles." }
    ],
    "temperature": 0.2,
    "max_tokens": 1024
  }'
```

### Tool calling (агенты в VSCode)

Сервер запускается с `--jinja`: llama.cpp использует Jinja2 chat template из GGUF
и отдаёт OpenAI-style function calling (`tools` в `/v1/chat/completions`).
Qwen2.5-Coder-32B поддерживает нативный формат **Hermes 2 Pro** — без `--jinja`
вызовы тулов попадают в текст ответа и агент их не выполняет.

Проверка после `docker compose up`:

```bash
curl http://localhost:8000/props | jq '.default_generation_settings.chat_template_caps'
# ожидайте tool_use / parallel_tool_calls в caps

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "qwen2.5-coder-32b",
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": { "city": { "type": "string" } },
          "required": ["city"]
        }
      }
    }],
    "messages": [{ "role": "user", "content": "What is the weather in Moscow?" }]
  }'
```

В ответе должен быть `tool_calls`, а не сырой XML в `content`.

### Подключение клиентов

- **Open WebUI** — уже настроен через `OPENAI_API_BASE_URL` в compose.
- **Continue** (Agent mode, `~/.continue/config.yaml`):

```yaml
name: Qwen V100
version: 0.0.1
schema: v1

models:
  - name: Qwen2.5-Coder-32B (V100)
    provider: openai
    model: qwen2.5-coder-32b
    apiBase: http://<host>:8000/v1
    apiKey: local-key
    capabilities:
      - tool_use
    roles:
      - chat
      - edit
      - apply
```

`capabilities: [tool_use]` нужен, чтобы Continue включил Agent mode и запуск
тулов (чтение файлов, терминал и т.д.) в VSCode.

- **Cline** — провайдер «OpenAI Compatible», Base URL `http://<host>:8000/v1`,
  API key `local-key`, model `qwen2.5-coder-32b`. Включите режим с tools/act mode;
  сервер уже отдаёт function calling через `--jinja`.

- **Aider**:

```bash
export OPENAI_API_BASE=http://<host>:8000/v1
export OPENAI_API_KEY=local-key
aider --model openai/qwen2.5-coder-32b
```

---

## Проверка мульти-GPU, NVLink и баланса VRAM

1. Обе карты заняты и VRAM сбалансирован (≈поровну):

```bash
watch -n 1 nvidia-smi
# во время инференса обе V100 показывают занятую память и периодическую загрузку
```

2. Логи llama-server подтверждают распределение по двум устройствам:

```bash
docker logs llama-server | grep -Ei "CUDA0|CUDA1|buffer size|offloaded"
# должны быть и CUDA0, и CUDA1 с примерно равными буферами весов/KV-cache
```

3. NVLink присутствует и используется:

```bash
nvidia-smi topo -m            # между GPU0 и GPU1 должно быть NV# (а не SYS/PHB)
nvidia-smi nvlink -s          # статус линков (Active)
# счётчики трафика по NVLink (растут при -sm row сильнее, чем при -sm layer):
nvidia-smi nvlink -gt d
```

4. CPU-offload отсутствует: в логах `offloaded 65/65 layers to GPU` и `-ngl 99` в команде.

---

## Тюнинг производительности

- **Split mode (`-sm`)** — главный рычаг под NVLink:
  - `layer` (по умолчанию) — деление по слоям, минимум межкарточного трафика,
    обычно лучший throughput для одного потока.
  - `row` — деление тензоров по строкам между GPU (ближе к tensor parallelism),
    активно использует NVLink, может снизить latency. Попробуйте:
    в `docker-compose.yml` замените `-sm layer` на `-sm row`.
- **Баланс VRAM (`-ts`)** — `1,1` делит поровну. Если одна карта подгружена
  чем-то ещё, сместите, например `-ts 6,4`.
- **Flash Attention + квантованный KV-cache** — экономит память KV и даёт более
  длинный контекст. На V100 включайте аккуратно (проверьте, что стартует):

```yaml
      - -fa
      - "on"
      - --cache-type-k
      - q8_0
      - --cache-type-v
      - q8_0
```

- **Квантизация модели** — `Q5_K_M` (качество, ~23 GB) ↔ `Q4_K_M` (скорость/запас, ~20 GB).
  Меняется через `MODEL_QUANT` в `.env`.
- **Контекст (`-c`)** — 32768 — нативный для модели. KV-cache для 32k влезает с запасом
  на 2×32 GB; при квантованном KV можно поднять до 65536.
- **Параллелизм (`-np`)** — для кодинг-агентов оставьте `1` (полный контекст одному
  запросу). Для нескольких пользователей увеличьте, но `-c` делится между слотами.
- **Запас памяти**: Q5_K_M (~23 GB весов) делится ≈11.5 GB на карту, остаётся
  ~20 GB на карту под KV-cache и буферы — это и позволяет CPU-offload не включать.
- Для разумных ответов в кодинге: `temperature` 0.1–0.3, `max_tokens` 1024–2048.

---

## Диагностика проблем

### Сборка падает / долгая

Первая сборка llama.cpp из исходников — это нормально несколько минут. Чтобы
зафиксировать версию и кэшировать, задайте тег в `.env`: `LLAMA_CPP_REF=b4400`.

### `CUDA driver version is insufficient`

Драйвер хоста старее, чем требует CUDA 12.1. Либо обновите драйвер (`>= 525.60.13`),
либо в `Dockerfile` замените оба `nvidia/cuda:12.1.0-*` на `nvidia/cuda:11.8.0-*`
и пересоберите (`docker compose build --no-cache`).

### `HTTPS is not supported` / `failed to load model ''`

В логах:

```
get_repo_commit: error: HTTPS is not supported. Please rebuild with -DLLAMA_OPENSSL=ON ...
llama_model_load_from_file_impl: exactly one out metadata, path_model, and file must be defined
load_model: failed to load model, ''
```

Это значит, что `llama-server` собран **без поддержки HTTPS**, поэтому `-hf` не может
скачать модель (путь к модели остаётся пустым → `''`). Лечится пересборкой: в `Dockerfile`
в build-стейдж добавлен `libssl-dev` и флаг `-DLLAMA_OPENSSL=ON`. Пересоберите без кэша:

```bash
docker compose build --no-cache
docker compose up -d
```

#### Запасной вариант: скачать GGUF вручную (без `-hf`)

Если по какой-то причине загрузка через `-hf` недоступна, скачайте GGUF на хост и
примонтируйте его как файл:

```bash
mkdir -p models
# пример для Q5_K_M (файл может быть разбит на части *-00001-of-0000N.gguf)
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o models/qwen2.5-coder-32b-q5_k_m.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q5_K_M.gguf"
```

Затем в `docker-compose.yml` примонтируйте папку и замените `-hf ...` на `-m`:

```yaml
    volumes:
      - ./models:/models
    command:
      - -m
      - /models/qwen2.5-coder-32b-q5_k_m.gguf
      # (для разбитой модели укажите первый файл *-00001-of-0000N.gguf)
      - --alias
      - ${MODEL_ALIAS}
      # ... остальные флаги без изменений
```

### Контейнер не видит GPU

Проверьте NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Если тут пусто/ошибка — переустановите toolkit и `sudo systemctl restart docker`.

### Проблемы с P2P/NVLink-копированием

llama.cpp не использует NCCL, но использует CUDA P2P. Если видите подвисания/ошибки
копий между GPU, отключите P2P (станет медленнее) — раскомментируйте в `docker-compose.yml`:

```yaml
      GGML_CUDA_NO_PEER_COPY: "1"
```

### Не хватает VRAM (OOM)

Уменьшите контекст (`-c 16384`), включите квантованный KV-cache (см. тюнинг),
или перейдите на `MODEL_QUANT=Q4_K_M`.

---

## Остановка

```bash
docker compose down          # volumes сохраняются (кэш модели + данные UI)
docker compose down -v       # также удалить volumes (перекачка модели заново)
```

---

## Вариант B (НЕ рекомендуется): остаться на vLLM

Если vLLM обязателен, нужен **старый** стек с поддержкой sm_70: образ с CUDA 11.8/12.1
и PyTorch, собранным с `sm_70` (например, vLLM `v0.4.x`–`v0.6.x` эпохи CUDA 12.1 +
torch 2.1–2.3). Избегайте `:latest`, nightly и сборок на CUDA 13. Это заметно
сложнее в поддержке и более хрупко, чем llama.cpp, поэтому рекомендуется Вариант A.
