# 상세페이지 이미지 다국어 번역 시스템

상품 상세페이지 이미지를 자동으로 다국어 번역하는 시스템입니다. Google Gemini AI를 활용한 이미지 번역과 PaddleOCR 기반 품질 검증을 제공합니다.

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                           n8n Cloud                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ 메인 워크플로우   │───▶│ 서브워크플로우    │───▶│ 배치 워커     │ │
│  │ (V7 Cloud)       │    │ (이미지 번역)     │    │ (결과 처리)   │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Text Render Service (Railway)                     │
├─────────────────────────────────────────────────────────────────────┤
│  • 이미지 슬라이싱/병합                                               │
│  • OCR 기반 번역 품질 검증 (PaddleOCR)                                │
│  • 배치 결과 처리                                                     │
│  • 재시도 큐 관리                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │  Supabase   │  │ Google AI   │  │   GCS       │
            │  (DB/Store) │  │ (Gemini)    │  │  (Storage)  │
            └─────────────┘  └─────────────┘  └─────────────┘
```

## 워크플로우 구성

### 1. 메인 워크플로우 (V7 Cloud)
**상세페이지 이미지 다국어 번역 (자동감지+언어선택+구글 이미지 배치기능)**

| 단계 | 설명 |
|------|------|
| 테이블 선택 | 폼 트리거로 번역할 테이블/언어 선택 |
| 설정 | 번역 설정 초기화 |
| 버킷 확인/생성 | GCS 스토리지 버킷 준비 |
| 상품 조회 | Supabase에서 미완료 상품 조회 |
| 이미지 분리 | 상품별 이미지 URL 분리 |
| 이미지 번역 실행 | 서브워크플로우 호출 |
| 번역 결과 처리 | 결과 검증 및 DB 업데이트 |

### 2. 서브워크플로우 (이미지 번역 처리)
개별 이미지 번역을 담당하는 워크플로우

- Gemini API를 통한 이미지 번역
- 슬라이싱/병합 처리 (대형 이미지)
- 번역 결과 반환

### 3. 배치 워커 (결과 처리)
**배치 워커 - 이미지 번역 결과 처리 (Cloud)**

| 단계 | 설명 |
|------|------|
| 5분마다 실행 | 스케줄 트리거 |
| 대기 작업 조회 | 완료 대기 중인 배치 작업 조회 |
| Gemini 상태 확인 | 배치 작업 완료 여부 확인 |
| 결과 처리 API 호출 | text-render-service로 결과 처리 요청 |
| 상품 업데이트 | Supabase DB 업데이트 |
| 이메일 알림 | 완료 알림 발송 |

## API 엔드포인트

### 핵심 API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서버 상태 확인 |
| `/validate-translation` | POST | OCR 기반 번역 품질 검증 |
| `/ocr-detect` | POST | OCR 텍스트 감지 |
| `/slice-image` | POST | 대형 이미지 슬라이싱 |
| `/merge-images` | POST | 번역된 이미지 조각 병합 |
| `/translate-chunks` | POST | 이미지 청크 번역 |

### 배치 처리 API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/prepare-batch` | POST | 배치 작업 준비 |
| `/process-batch-results` | POST | 배치 결과 처리 |

### 재시도 큐 API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/retry-queue/add` | POST | 재시도 큐에 작업 추가 |
| `/retry-queue/list` | GET/POST | 재시도 큐 목록 조회 |
| `/retry-queue/process` | POST | 재시도 큐 처리 |
| `/retry-queue/check-and-merge` | POST | 재시도 결과 확인 및 병합 |

### 청크 관리 API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/get-original-chunk` | POST | 원본 청크 조회 |
| `/cleanup-original-chunks` | POST | 원본 청크 정리 |

## OCR 번역 검증

PaddleOCR을 사용하여 번역된 이미지의 품질을 자동 검증합니다.

### 검증 로직
1. 번역된 이미지에서 텍스트 추출 (PaddleOCR 3.x)
2. 타겟 언어 비율 계산
3. 비타겟 언어가 20% 이상이면 번역 미완료로 판정

### 지원 언어
| 코드 | 언어 |
|------|------|
| en | 영어 |
| ja | 일본어 |
| zh-CN | 중국어 간체 |
| zh-TW | 중국어 번체 |
| ko | 한국어 |
| vi | 베트남어 |
| th | 태국어 |
| id | 인도네시아어 |

### 검증 응답 예시
```json
{
  "valid": true,
  "reason": "번역 검증 통과: 타겟 언어(en) 비율 93.3%",
  "has_text": true,
  "total_chars": 817,
  "target_lang_ratio": 0.933,
  "detected_text": [
    {"text": "PURPOSE", "confidence": 0.999, "char_count": 7},
    {"text": "HOW TO USE", "confidence": 0.999, "char_count": 8}
  ]
}
```

## 기술 스택

| 구성요소 | 기술 |
|---------|------|
| 워크플로우 | n8n Cloud |
| 백엔드 API | Python Flask + Gunicorn |
| OCR 엔진 | PaddleOCR 3.x (paddlepaddle 3.2.0, paddleocr 3.3.3) |
| AI 번역 | Google Gemini API |
| 데이터베이스 | Supabase (PostgreSQL) |
| 스토리지 | Google Cloud Storage |
| 호스팅 | Railway |

## 설치 및 실행

### 로컬 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python server.py

# 또는 Gunicorn으로 실행
gunicorn --bind 0.0.0.0:5001 --workers 2 --timeout 300 server:app
```

### Docker 실행
```bash
docker build -t text-render-service .
docker run -p 5001:5001 text-render-service
```

## 환경 변수

```env
PORT=5001
WORKERS=2
PYTHONUNBUFFERED=1

# PaddlePaddle 호환성 설정
FLAGS_enable_pir_api=0
FLAGS_enable_pir_in_executor=0
FLAGS_use_mkldnn=0
```

## 배포

### Railway 자동 배포
GitHub main 브랜치에 푸시하면 자동으로 Railway에 배포됩니다.

```bash
git add .
git commit -m "feat: add new feature"
git push origin main
```

## 성능 최적화

### OCR 처리 최적화
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `text_det_limit_side_len` | 736 | 이미지 최대 변 길이 제한 |
| `cpu_threads` | 8 | CPU 스레드 수 |
| `enable_mkldnn` | True | Intel MKL-DNN 가속 |
| `text_det_box_thresh` | 0.5 | 텍스트 박스 임계값 |
| `text_rec_score_thresh` | 0.3 | 인식 점수 임계값 |

### 처리 성능
| 이미지 크기 | 처리 시간 | 정확도 |
|------------|----------|--------|
| 850x1327px | ~14초 | 95-99% |

## 라이선스

Private - All rights reserved
