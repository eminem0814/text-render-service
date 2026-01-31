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

# Python 의존성 설치 (PaddleOCR - 경량화된 OCR)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 애플리케이션 코드 복사
COPY server.py batch_processor.py ./

# 환경 변수
ENV PORT=5001
ENV WORKERS=2
ENV PYTHONUNBUFFERED=1
# PaddlePaddle 호환성 설정
ENV FLAGS_enable_pir_api=0
ENV FLAGS_enable_pir_in_executor=0
ENV FLAGS_use_mkldnn=0

# 헬스체크 (PaddleOCR 초기화 시간 고려)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Gunicorn으로 프로덕션 실행 (타임아웃 10분)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers ${WORKERS} --timeout 600 --graceful-timeout 600 --keep-alive 5 server:app"]
