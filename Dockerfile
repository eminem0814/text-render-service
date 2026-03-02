FROM python:3.11-slim

# 시스템 의존성 설치 (OpenCV, PaddleOCR, 폰트 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-cjk \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성 설치 (PaddleOCR + FastText)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# FastText 언어 감지 모델 다운로드 (압축 모델 917KB)
RUN mkdir -p /app/models && \
    curl -L -o /app/models/lid.176.ftz \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

# 애플리케이션 코드 복사
COPY server.py batch_processor.py ./

# 환경 변수
ENV PORT=5001
ENV WORKERS=1
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=65536
# PaddlePaddle 호환성 설정
ENV FLAGS_enable_pir_api=0
ENV FLAGS_enable_pir_in_executor=0
ENV FLAGS_use_mkldnn=0
# PaddleOCR 모델 소스 체크 비활성화 (콜드 스타트 지연 방지)
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
# 서버 시작 시 워밍업할 OCR 언어 (쉼표 구분)
ENV OCR_WARMUP_LANGS=ko,en
# FastText 모델 경로
ENV FASTTEXT_MODEL_PATH=/app/models/lid.176.ftz

# 헬스체크 (PaddleOCR 초기화 시간 고려)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Gunicorn으로 프로덕션 실행 (타임아웃 10분)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers ${WORKERS} --timeout 600 --graceful-timeout 600 --keep-alive 5 server:app"]
