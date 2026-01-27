# 이미지 텍스트 렌더링 서비스

n8n 이미지 번역 워크플로우와 함께 사용하는 텍스트 삽입 서비스입니다.

## 설치

```bash
cd /Users/johnkim/N8N/scraper-api/text-render-service
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
python server.py
```

서버가 `http://localhost:5000`에서 실행됩니다.

## API 엔드포인트

### 헬스 체크
```
GET /health
```

### 텍스트 렌더링
```
POST /render-text
Content-Type: application/json

{
  "image_url": "https://example.com/cleaned-image.jpg",
  "text_blocks": [
    {
      "text": "원본 한글",
      "translatedText": "Translated English",
      "bounds": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 30
      }
    }
  ]
}
```

## 필요한 API 키

워크플로우를 사용하려면 다음 API 키가 필요합니다:

### 1. Google Cloud API Key
- [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
- Cloud Vision API 활성화
- Cloud Translation API 활성화
- API 키 생성

### 2. Replicate API Token
- [Replicate](https://replicate.com/)에서 계정 생성
- API Token 발급

## 워크플로우 설정

`workflow-image-translate-v2.json`의 "설정" 노드에서 API 키를 입력하세요:

```json
{
  "googleVisionApiKey": "YOUR_GOOGLE_CLOUD_API_KEY",
  "replicateApiToken": "YOUR_REPLICATE_API_TOKEN",
  "textServiceUrl": "http://localhost:5000/render-text"
}
```

## 파이프라인 흐름

```
1. 이미지 다운로드
2. Google Vision API → 한글 텍스트 위치 + 내용 감지
3. Google Translate → 한글 → 영어 번역
4. LaMa (Replicate) → 한글 텍스트 영역 제거 (인페인팅)
5. 텍스트 렌더링 서비스 → 영어 텍스트 삽입
6. Supabase 저장
```

## 폰트 커스터마이징

`server.py`의 `FONT_PATHS`를 수정하여 원하는 폰트를 사용할 수 있습니다:

```python
FONT_PATHS = [
    "/path/to/your/font.ttf",
    # ...
]
```
