"""
텍스트 번역 및 렌더링 서비스 v9
- 긴 이미지 슬라이스/병합 지원
- Gemini 배치 처리 결과 처리
- 청크 번역 및 검증
- OCR 기반 번역 언어 검증
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
import base64
import os
import logging
import json

# PaddlePaddle PIR 비활성화 (호환성 문제 해결)
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'

from paddleocr import PaddleOCR

app = Flask(__name__)

# ===== 성능 최적화 설정 =====
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_DIMENSION = 30000  # 30000px
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== OCR 설정 (PaddleOCR) =====
# PaddleOCR 언어 코드 매핑 (시스템 언어코드 -> PaddleOCR 언어코드)
LANGUAGE_CODE_MAP = {
    # 동아시아
    "ko": "korean",      # 한국어
    "ja": "japan",       # 일본어
    "zh": "ch",          # 중국어 간체
    "zh-CN": "ch",
    "zh-TW": "chinese_cht",  # 중국어 번체
    # 유럽
    "en": "en",          # 영어
    "de": "german",      # 독일어
    "fr": "french",      # 프랑스어
    "es": "es",          # 스페인어
    "it": "it",          # 이탈리아어
    "pt": "pt",          # 포르투갈어
    "ru": "ru",          # 러시아어
    # 동남아
    "th": "th",          # 태국어
    "vi": "vi",          # 베트남어
    "id": "id",          # 인도네시아어
    # 기타
    "ar": "ar",          # 아랍어
    "hi": "hi",          # 힌디어
}

# OCR Reader 캐시 (언어별로 생성하여 재사용)
ocr_readers = {}

def get_ocr_reader(target_lang: str):
    """
    대상 언어에 맞는 PaddleOCR Reader를 반환 (캐싱)

    Args:
        target_lang: 대상 언어 코드 (예: 'en', 'ja', 'zh')

    Returns:
        PaddleOCR 객체
    """
    paddle_lang = LANGUAGE_CODE_MAP.get(target_lang, "en")

    if paddle_lang not in ocr_readers:
        try:
            # PaddleOCR 3.x 초기화
            # 참고: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html
            import logging
            logging.getLogger('ppocr').setLevel(logging.WARNING)

            ocr_readers[paddle_lang] = PaddleOCR(
                lang=paddle_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # CPU 최적화
                cpu_threads=8,
                enable_mkldnn=True,
                # 대형 이미지 처리 - 내부적으로 최대 736px로 리사이즈 (속도 향상)
                text_det_limit_side_len=736,
                text_det_limit_type='max',
                # 텍스트 감지 민감도 조정
                text_det_box_thresh=0.5,
                text_det_thresh=0.3,
                text_det_unclip_ratio=1.6,
                text_rec_score_thresh=0.3,
            )
            logger.info(f"PaddleOCR Reader 생성: {paddle_lang}")
        except Exception as e:
            logger.error(f"PaddleOCR Reader 생성 실패: {e}")
            # 실패시 영어 기본 리더 반환
            if "en" not in ocr_readers:
                ocr_readers["en"] = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    cpu_threads=8,
                    enable_mkldnn=True,
                    text_det_limit_side_len=736,
                    text_det_limit_type='max',
                    text_det_box_thresh=0.5,
                    text_det_thresh=0.3,
                    text_det_unclip_ratio=1.6,
                    text_rec_score_thresh=0.3,
                )
            return ocr_readers["en"]

    return ocr_readers[paddle_lang]


def validate_translation_language(image_base64: str, target_lang: str, threshold: float = 0.2) -> dict:
    """
    OCR을 사용하여 번역된 이미지가 올바른 언어인지 검증

    Args:
        image_base64: 검증할 이미지의 base64 문자열
        target_lang: 번역 대상 언어 코드 (예: 'en', 'ja', 'zh')
        threshold: 비타겟 언어 허용 비율 (기본 20%)

    Returns:
        dict: {
            "valid": bool,          # 검증 통과 여부
            "reason": str,          # 결과 사유
            "has_text": bool,       # 텍스트 존재 여부
            "total_chars": int,     # 총 감지된 문자 수
            "target_lang_ratio": float,  # 타겟 언어 비율
            "detected_text": list   # 감지된 텍스트 목록 (디버깅용)
        }
    """
    try:
        # 1. 이미지 디코딩
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))

        # PIL Image를 numpy array로 변환
        image_np = np.array(image)

        # RGB 변환 (필요한 경우)
        if len(image_np.shape) == 2:  # 그레이스케일
            pass  # PaddleOCR은 그레이스케일도 처리 가능
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # 2. OCR 실행 (PaddleOCR 3.x - predict 메서드 사용)
        reader = get_ocr_reader(target_lang)
        results = reader.predict(image_np)

        # 3. 텍스트 없으면 통과 (예외 케이스)
        if results is None:
            return {
                "valid": True,
                "reason": "텍스트 없음 - 검증 통과",
                "has_text": False,
                "total_chars": 0,
                "target_lang_ratio": 1.0,
                "detected_text": []
            }

        # 4. 감지된 텍스트 분석 (PaddleOCR 3.x 형식)
        detected_texts = []
        total_chars = 0

        # PaddleOCR 3.x: results는 이터레이터, 각 res는 json 속성 보유
        try:
            for res in results:
                # PaddleOCR 3.x: res.json['res'] 안에 실제 결과가 있음
                inner_res = res.json.get('res', {}) if hasattr(res.json, 'get') else {}
                rec_texts = inner_res.get('rec_texts', []) if hasattr(inner_res, 'get') else []
                rec_scores = inner_res.get('rec_scores', []) if hasattr(inner_res, 'get') else []

                for text, confidence in zip(rec_texts, rec_scores):
                    if confidence is None or confidence <= 0.3:
                        continue
                    # 신뢰도 30% 이상만
                    detected_texts.append({
                        "text": str(text),
                        "confidence": float(confidence),
                        "char_count": len(str(text).replace(" ", ""))
                    })
                    total_chars += len(str(text).replace(" ", ""))
        except Exception as parse_error:
            logger.error(f"OCR 결과 파싱 오류: {parse_error}, results type: {type(results)}")
            return {
                "valid": True,
                "reason": f"OCR 파싱 오류로 통과 처리: {str(parse_error)}",
                "has_text": False,
                "total_chars": 0,
                "target_lang_ratio": 1.0,
                "detected_text": []
            }

        # 텍스트가 너무 적으면 통과
        if total_chars < 5:
            return {
                "valid": True,
                "reason": "텍스트 적음 - 검증 통과",
                "has_text": True,
                "total_chars": total_chars,
                "target_lang_ratio": 1.0,
                "detected_text": detected_texts
            }

        # 5. 언어 판별 (간단한 휴리스틱)
        target_lang_chars = count_target_language_chars(
            "".join([t["text"] for t in detected_texts]),
            target_lang
        )

        target_ratio = target_lang_chars / total_chars if total_chars > 0 else 1.0
        non_target_ratio = 1.0 - target_ratio

        # 6. 검증 결과
        if non_target_ratio > threshold:
            return {
                "valid": False,
                "reason": f"번역 미완료: 타겟 언어({target_lang}) 비율 {target_ratio:.1%}, 비타겟 언어 비율 {non_target_ratio:.1%}",
                "has_text": True,
                "total_chars": total_chars,
                "target_lang_ratio": target_ratio,
                "detected_text": detected_texts
            }

        return {
            "valid": True,
            "reason": f"번역 검증 통과: 타겟 언어({target_lang}) 비율 {target_ratio:.1%}",
            "has_text": True,
            "total_chars": total_chars,
            "target_lang_ratio": target_ratio,
            "detected_text": detected_texts
        }

    except Exception as e:
        logger.error(f"번역 언어 검증 오류: {e}")
        # 오류 발생시 통과 처리 (false positive 방지)
        return {
            "valid": True,
            "reason": f"검증 오류로 통과 처리: {str(e)}",
            "has_text": False,
            "total_chars": 0,
            "target_lang_ratio": 1.0,
            "detected_text": []
        }


def count_target_language_chars(text: str, target_lang: str) -> int:
    """
    텍스트에서 타겟 언어의 문자 수를 카운트

    Args:
        text: 분석할 텍스트
        target_lang: 타겟 언어 코드

    Returns:
        int: 타겟 언어 문자 수
    """
    count = 0

    for char in text:
        if char.isspace():
            continue

        code = ord(char)

        # 한국어 (Hangul)
        if target_lang == "ko":
            if (0xAC00 <= code <= 0xD7AF or  # 한글 음절
                0x1100 <= code <= 0x11FF or  # 한글 자모
                0x3130 <= code <= 0x318F):   # 호환 자모
                count += 1

        # 일본어 (Hiragana, Katakana, 일부 한자)
        elif target_lang == "ja":
            if (0x3040 <= code <= 0x309F or  # 히라가나
                0x30A0 <= code <= 0x30FF or  # 가타카나
                0x4E00 <= code <= 0x9FFF):   # CJK 한자
                count += 1

        # 중국어 (한자)
        elif target_lang in ["zh", "zh-CN", "zh-TW"]:
            if 0x4E00 <= code <= 0x9FFF:  # CJK 한자
                count += 1

        # 영어/유럽어 (라틴 문자)
        elif target_lang in ["en", "de", "fr", "es", "it", "pt"]:
            if (0x0041 <= code <= 0x005A or  # A-Z
                0x0061 <= code <= 0x007A or  # a-z
                0x00C0 <= code <= 0x00FF):   # 확장 라틴
                count += 1

        # 러시아어 (키릴 문자)
        elif target_lang == "ru":
            if 0x0400 <= code <= 0x04FF:  # 키릴 문자
                count += 1

        # 태국어
        elif target_lang == "th":
            if 0x0E00 <= code <= 0x0E7F:  # 태국 문자
                count += 1

        # 베트남어 (라틴 + 성조)
        elif target_lang == "vi":
            if (0x0041 <= code <= 0x005A or
                0x0061 <= code <= 0x007A or
                0x00C0 <= code <= 0x024F or  # 확장 라틴
                0x1EA0 <= code <= 0x1EFF):   # 베트남어 특수문자
                count += 1

        # 아랍어
        elif target_lang == "ar":
            if 0x0600 <= code <= 0x06FF:  # 아랍 문자
                count += 1

        # 기본: 숫자와 특수문자는 중립으로 처리
        elif char.isdigit() or not char.isalnum():
            count += 1  # 숫자/특수문자는 타겟 언어로 간주

    return count


# Google Cloud 설정 (선택사항 - 슬라이스/병합에는 불필요)
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Supabase 설정 (환경변수에서 가져옴 - 배치 처리용)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# 원본 청크 저장 버킷 (재처리용)
ORIGINAL_CHUNKS_BUCKET = "original-chunks"


# ===== 원본 청크 저장/조회 함수 =====

def save_original_chunk(
    chunk_base64: str,
    chunk_key: str,
    batch_id: str,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    원본 청크를 Supabase Storage에 저장

    Args:
        chunk_base64: 원본 청크 이미지의 base64 문자열
        chunk_key: 청크 식별자 (예: "p123_i0_c0")
        batch_id: 배치 작업 ID (폴더 구분용)
        supabase_url: Supabase URL (없으면 환경변수 사용)
        supabase_key: Supabase 키 (없으면 환경변수 사용)

    Returns:
        dict: {
            "success": bool,
            "path": str (저장 경로),
            "error": str (에러 메시지)
        }
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "path": "", "error": "Supabase 설정 없음"}

    try:
        # 저장 경로: original-chunks/{batch_id}/{chunk_key}.jpg
        storage_path = f"{batch_id}/{chunk_key}.jpg"

        # base64 디코딩
        image_bytes = base64.b64decode(chunk_base64)

        # Supabase Storage에 업로드
        upload_url = f"{url}/storage/v1/object/{ORIGINAL_CHUNKS_BUCKET}/{storage_path}"

        response = requests.post(
            upload_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "image/jpeg"
            },
            data=image_bytes,
            timeout=30
        )

        if response.status_code in [200, 201]:
            return {"success": True, "path": storage_path, "error": ""}
        else:
            # 이미 존재하면 덮어쓰기 시도
            if response.status_code == 400 and "already exists" in response.text.lower():
                # upsert로 재시도
                response = requests.put(
                    upload_url,
                    headers={
                        "apikey": key,
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "image/jpeg"
                    },
                    data=image_bytes,
                    timeout=30
                )
                if response.status_code in [200, 201]:
                    return {"success": True, "path": storage_path, "error": ""}

            return {
                "success": False,
                "path": "",
                "error": f"업로드 실패: {response.status_code} - {response.text[:200]}"
            }

    except Exception as e:
        return {"success": False, "path": "", "error": str(e)}


def get_original_chunk(
    chunk_key: str,
    batch_id: str,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    Supabase Storage에서 원본 청크 조회

    Args:
        chunk_key: 청크 식별자 (예: "p123_i0_c0")
        batch_id: 배치 작업 ID
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {
            "success": bool,
            "base64": str (원본 이미지 base64),
            "error": str
        }
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "base64": "", "error": "Supabase 설정 없음"}

    try:
        storage_path = f"{batch_id}/{chunk_key}.jpg"
        download_url = f"{url}/storage/v1/object/{ORIGINAL_CHUNKS_BUCKET}/{storage_path}"

        response = requests.get(
            download_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}"
            },
            timeout=30
        )

        if response.status_code == 200:
            chunk_base64 = base64.b64encode(response.content).decode('utf-8')
            return {"success": True, "base64": chunk_base64, "error": ""}
        else:
            return {
                "success": False,
                "base64": "",
                "error": f"다운로드 실패: {response.status_code}"
            }

    except Exception as e:
        return {"success": False, "base64": "", "error": str(e)}


def delete_original_chunks(
    batch_id: str,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    배치 작업의 원본 청크들을 일괄 삭제 (정리용)

    Args:
        batch_id: 배치 작업 ID
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {"success": bool, "deleted_count": int, "error": str}
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "deleted_count": 0, "error": "Supabase 설정 없음"}

    try:
        # 배치 폴더의 파일 목록 조회
        list_url = f"{url}/storage/v1/object/list/{ORIGINAL_CHUNKS_BUCKET}"

        response = requests.post(
            list_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={"prefix": f"{batch_id}/"},
            timeout=30
        )

        if response.status_code != 200:
            return {"success": False, "deleted_count": 0, "error": f"목록 조회 실패: {response.status_code}"}

        files = response.json()
        if not files:
            return {"success": True, "deleted_count": 0, "error": ""}

        # 파일들 삭제
        file_paths = [f"{batch_id}/{f['name']}" for f in files if f.get('name')]

        if not file_paths:
            return {"success": True, "deleted_count": 0, "error": ""}

        delete_url = f"{url}/storage/v1/object/{ORIGINAL_CHUNKS_BUCKET}"
        delete_response = requests.delete(
            delete_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={"prefixes": file_paths},
            timeout=60
        )

        if delete_response.status_code in [200, 204]:
            return {"success": True, "deleted_count": len(file_paths), "error": ""}
        else:
            return {
                "success": False,
                "deleted_count": 0,
                "error": f"삭제 실패: {delete_response.status_code}"
            }

    except Exception as e:
        return {"success": False, "deleted_count": 0, "error": str(e)}


# ===== 재처리 대기열 관리 함수 =====

# 재처리 대기열 테이블명
RETRY_QUEUE_TABLE = "chunk_retry_queue"
# 청크 결과 테이블명 (병합 트리거용)
CHUNK_RESULTS_TABLE = "chunk_results"


def enqueue_failed_chunk(
    chunk_info: dict,
    config: dict,
    error_reason: str,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    실패한 청크를 재처리 대기열에 추가

    Args:
        chunk_info: 청크 정보 {
            batch_id, chunk_key, product_id, image_index, chunk_index,
            total_chunks, chunk_width, chunk_height, original_chunk_path
        }
        config: 처리 설정 {target_lang, prompt, gemini_model, ...}
        error_reason: 실패 사유
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {"success": bool, "queue_id": int, "error": str}
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "queue_id": None, "error": "Supabase 설정 없음"}

    try:
        insert_url = f"{url}/rest/v1/{RETRY_QUEUE_TABLE}"

        queue_item = {
            "batch_id": chunk_info.get("batch_id"),
            "chunk_key": chunk_info.get("chunk_key"),
            "product_id": chunk_info.get("product_id"),
            "product_code": chunk_info.get("product_code"),
            "image_index": chunk_info.get("image_index"),
            "chunk_index": chunk_info.get("chunk_index"),
            "total_chunks": chunk_info.get("total_chunks"),
            "chunk_width": chunk_info.get("chunk_width"),
            "chunk_height": chunk_info.get("chunk_height"),
            "original_chunk_path": chunk_info.get("original_chunk_path"),
            "target_lang": config.get("targetLangCode", "en"),
            "prompt": config.get("prompt", ""),
            "gemini_model": config.get("geminiModel", "gemini-2.0-flash-001"),
            "gemini_api_key": config.get("geminiApiKey", ""),
            "storage_bucket": config.get("storageBucket", "translated-images"),
            "table_name": config.get("tableName", ""),
            "output_format": config.get("outputFormat", "WEBP"),
            "output_quality": config.get("outputQuality", 100),
            "status": "pending",
            "retry_count": 0,
            "last_error": error_reason,
            "config_json": json.dumps(config)
        }

        response = requests.post(
            insert_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            },
            json=queue_item,
            timeout=30
        )

        if response.status_code in [200, 201]:
            result = response.json()
            queue_id = result[0].get("id") if result else None
            logger.info(f"[대기열 추가] {chunk_info.get('chunk_key')}: queue_id={queue_id}")
            return {"success": True, "queue_id": queue_id, "error": ""}
        else:
            return {
                "success": False,
                "queue_id": None,
                "error": f"대기열 추가 실패: {response.status_code} - {response.text[:200]}"
            }

    except Exception as e:
        return {"success": False, "queue_id": None, "error": str(e)}


def get_pending_queue_items(
    batch_id: str = None,
    limit: int = 50,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    대기 중인 재처리 항목 조회

    Args:
        batch_id: 특정 배치만 조회 (없으면 전체)
        limit: 최대 조회 수
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {"success": bool, "items": list, "error": str}
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "items": [], "error": "Supabase 설정 없음"}

    try:
        query_url = f"{url}/rest/v1/{RETRY_QUEUE_TABLE}?status=eq.pending&order=created_at&limit={limit}"

        if batch_id:
            query_url += f"&batch_id=eq.{batch_id}"

        response = requests.get(
            query_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}"
            },
            timeout=30
        )

        if response.status_code == 200:
            return {"success": True, "items": response.json(), "error": ""}
        else:
            return {
                "success": False,
                "items": [],
                "error": f"조회 실패: {response.status_code}"
            }

    except Exception as e:
        return {"success": False, "items": [], "error": str(e)}


def update_queue_item_status(
    queue_id: int,
    status: str,
    error_msg: str = None,
    translated_base64: str = None,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    대기열 항목 상태 업데이트

    Args:
        queue_id: 대기열 ID
        status: 새 상태 (processing, completed, failed)
        error_msg: 에러 메시지 (실패 시)
        translated_base64: 번역된 이미지 base64 (성공 시)
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {"success": bool, "error": str}
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "error": "Supabase 설정 없음"}

    try:
        update_url = f"{url}/rest/v1/{RETRY_QUEUE_TABLE}?id=eq.{queue_id}"

        update_data = {
            "status": status,
            "updated_at": "now()"
        }

        if status == "processing":
            update_data["retry_count"] = "retry_count + 1"  # SQL expression won't work here

        if error_msg:
            update_data["last_error"] = error_msg

        if translated_base64:
            update_data["translated_base64"] = translated_base64

        response = requests.patch(
            update_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json=update_data,
            timeout=30
        )

        if response.status_code in [200, 204]:
            return {"success": True, "error": ""}
        else:
            return {"success": False, "error": f"업데이트 실패: {response.status_code}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def save_chunk_result(
    chunk_info: dict,
    translated_base64: str,
    status: str = "completed",
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    청크 처리 결과 저장 (병합 트리거용)

    Args:
        chunk_info: 청크 정보
        translated_base64: 번역된 이미지 base64
        status: 상태 (completed, failed)
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {"success": bool, "result_id": int, "error": str}
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"success": False, "result_id": None, "error": "Supabase 설정 없음"}

    try:
        # upsert를 위해 unique key 사용
        upsert_url = f"{url}/rest/v1/{CHUNK_RESULTS_TABLE}"

        result_item = {
            "batch_id": chunk_info.get("batch_id"),
            "chunk_key": chunk_info.get("chunk_key"),
            "product_id": chunk_info.get("product_id"),
            "product_code": chunk_info.get("product_code"),
            "image_index": chunk_info.get("image_index"),
            "chunk_index": chunk_info.get("chunk_index"),
            "total_chunks": chunk_info.get("total_chunks"),
            "chunk_width": chunk_info.get("chunk_width"),
            "chunk_height": chunk_info.get("chunk_height"),
            "translated_base64": translated_base64,
            "status": status
        }

        response = requests.post(
            upsert_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation,resolution=merge-duplicates"
            },
            json=result_item,
            timeout=30
        )

        if response.status_code in [200, 201]:
            result = response.json()
            result_id = result[0].get("id") if result else None
            return {"success": True, "result_id": result_id, "error": ""}
        else:
            return {
                "success": False,
                "result_id": None,
                "error": f"저장 실패: {response.status_code} - {response.text[:200]}"
            }

    except Exception as e:
        return {"success": False, "result_id": None, "error": str(e)}


def check_image_completion(
    batch_id: str,
    product_id: int,
    image_index: int,
    total_chunks: int,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    특정 이미지의 모든 청크가 완료되었는지 확인

    Args:
        batch_id: 배치 ID
        product_id: 상품 ID
        image_index: 이미지 인덱스
        total_chunks: 총 청크 수
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {
            "complete": bool,
            "completed_count": int,
            "total_chunks": int,
            "chunks": list (완료된 경우 청크 데이터 포함)
        }
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    if not url or not key:
        return {"complete": False, "completed_count": 0, "total_chunks": total_chunks, "chunks": []}

    try:
        # 해당 이미지의 완료된 청크 조회
        query_url = (
            f"{url}/rest/v1/{CHUNK_RESULTS_TABLE}"
            f"?batch_id=eq.{batch_id}"
            f"&product_id=eq.{product_id}"
            f"&image_index=eq.{image_index}"
            f"&status=eq.completed"
            f"&order=chunk_index"
        )

        response = requests.get(
            query_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}"
            },
            timeout=30
        )

        if response.status_code == 200:
            chunks = response.json()
            completed_count = len(chunks)
            is_complete = completed_count >= total_chunks

            return {
                "complete": is_complete,
                "completed_count": completed_count,
                "total_chunks": total_chunks,
                "chunks": chunks if is_complete else []
            }
        else:
            return {
                "complete": False,
                "completed_count": 0,
                "total_chunks": total_chunks,
                "chunks": [],
                "error": f"조회 실패: {response.status_code}"
            }

    except Exception as e:
        return {
            "complete": False,
            "completed_count": 0,
            "total_chunks": total_chunks,
            "chunks": [],
            "error": str(e)
        }


def process_single_retry_item(
    item: dict,
    supabase_url: str = None,
    supabase_key: str = None
) -> dict:
    """
    단일 재처리 항목 처리

    Args:
        item: 대기열 항목
        supabase_url: Supabase URL
        supabase_key: Supabase 키

    Returns:
        dict: {
            "success": bool,
            "translated_base64": str,
            "error": str,
            "image_complete": bool
        }
    """
    url = supabase_url or SUPABASE_URL
    key = supabase_key or SUPABASE_SERVICE_KEY

    queue_id = item.get("id")
    chunk_key = item.get("chunk_key")
    batch_id = item.get("batch_id")

    logger.info(f"[재처리 시작] {chunk_key}")

    # 1. 상태를 processing으로 변경
    update_queue_item_status(queue_id, "processing", supabase_url=url, supabase_key=key)

    try:
        # 2. 원본 청크 가져오기
        original_result = get_original_chunk(
            chunk_key=chunk_key,
            batch_id=batch_id,
            supabase_url=url,
            supabase_key=key
        )

        if not original_result["success"]:
            error_msg = f"원본 청크 조회 실패: {original_result['error']}"
            update_queue_item_status(queue_id, "failed", error_msg=error_msg, supabase_url=url, supabase_key=key)
            return {"success": False, "translated_base64": "", "error": error_msg, "image_complete": False}

        original_base64 = original_result["base64"]

        # 3. 실시간 API로 번역 시도
        gemini_api_key = item.get("gemini_api_key")
        prompt = item.get("prompt", "Translate the text in this image.")
        gemini_model = item.get("gemini_model", "gemini-2.0-flash-001")

        retry_result = retry_chunk_realtime(
            chunk_base64=original_base64,
            gemini_api_key=gemini_api_key,
            prompt=prompt,
            gemini_model=gemini_model,
            max_retries=3
        )

        if not retry_result["success"]:
            error_msg = f"번역 실패: {retry_result['error']}"
            update_queue_item_status(queue_id, "failed", error_msg=error_msg, supabase_url=url, supabase_key=key)
            return {"success": False, "translated_base64": "", "error": error_msg, "image_complete": False}

        translated_base64 = retry_result["base64"]

        # 4. 번역 결과 검증
        target_lang = item.get("target_lang", "en")
        validation = validate_translated_chunk(
            chunk_base64=translated_base64,
            target_lang=target_lang,
            expected_width=item.get("chunk_width"),
            expected_height=item.get("chunk_height"),
            size_tolerance=0.3,
            lang_threshold=0.2
        )

        if not validation["valid"]:
            error_msg = f"검증 실패: {validation['reason']}"
            # 재시도 횟수가 3회 미만이면 다시 pending으로
            retry_count = item.get("retry_count", 0)
            if retry_count < 3:
                update_queue_item_status(queue_id, "pending", error_msg=error_msg, supabase_url=url, supabase_key=key)
            else:
                update_queue_item_status(queue_id, "failed", error_msg=error_msg, supabase_url=url, supabase_key=key)
            return {"success": False, "translated_base64": "", "error": error_msg, "image_complete": False}

        # 5. 성공 - 대기열 상태 업데이트
        update_queue_item_status(
            queue_id, "completed",
            translated_base64=translated_base64,
            supabase_url=url, supabase_key=key
        )

        # 6. 청크 결과 저장
        chunk_info = {
            "batch_id": batch_id,
            "chunk_key": chunk_key,
            "product_id": item.get("product_id"),
            "product_code": item.get("product_code"),
            "image_index": item.get("image_index"),
            "chunk_index": item.get("chunk_index"),
            "total_chunks": item.get("total_chunks"),
            "chunk_width": item.get("chunk_width"),
            "chunk_height": item.get("chunk_height")
        }

        save_chunk_result(chunk_info, translated_base64, "completed", supabase_url=url, supabase_key=key)

        # 7. 이미지 완료 여부 확인
        completion = check_image_completion(
            batch_id=batch_id,
            product_id=item.get("product_id"),
            image_index=item.get("image_index"),
            total_chunks=item.get("total_chunks"),
            supabase_url=url,
            supabase_key=key
        )

        logger.info(f"[재처리 성공] {chunk_key}: 이미지 완료={completion['complete']}")

        return {
            "success": True,
            "translated_base64": translated_base64,
            "error": "",
            "image_complete": completion["complete"],
            "completion_info": completion
        }

    except Exception as e:
        error_msg = f"처리 오류: {str(e)}"
        update_queue_item_status(queue_id, "failed", error_msg=error_msg, supabase_url=url, supabase_key=key)
        return {"success": False, "translated_base64": "", "error": error_msg, "image_complete": False}


# Vertex AI 초기화 시도 (선택 사항)
vertex_ai_available = False
try:
    if os.path.exists(CREDENTIALS_PATH):
        with open(CREDENTIALS_PATH, 'r') as f:
            creds_data = json.load(f)
            if isinstance(creds_data, dict) and creds_data.get("type") == "service_account":
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
                PROJECT_ID = creds_data.get("project_id", "")
                import vertexai
                vertexai.init(project=PROJECT_ID, location=LOCATION)
                vertex_ai_available = True
                logger.info(f"Vertex AI initialized: project={PROJECT_ID}")
except Exception as e:
    logger.warning(f"Vertex AI not available: {e}")


def validate_chunk(chunk_base64, expected_width=None, expected_height=None, tolerance=0.3):
    """
    청크 이미지 검증

    Args:
        chunk_base64: 검증할 이미지의 base64 문자열
        expected_width: 예상 너비 (없으면 검증 스킵)
        expected_height: 예상 높이 (없으면 검증 스킵)
        tolerance: 허용 오차 비율 (기본 30%)

    Returns:
        dict: {
            "valid": bool,
            "reason": str (실패 사유),
            "actual_width": int,
            "actual_height": int
        }
    """
    try:
        # 1. Base64 디코딩 가능 여부
        if not chunk_base64 or len(chunk_base64) < 1000:
            return {"valid": False, "reason": "이미지 데이터 없음 또는 너무 작음", "actual_width": 0, "actual_height": 0}

        # 2. 이미지 디코딩
        try:
            image_bytes = base64.b64decode(chunk_base64)
            image = Image.open(BytesIO(image_bytes))
            actual_width, actual_height = image.size
        except Exception as e:
            return {"valid": False, "reason": f"이미지 디코딩 실패: {str(e)}", "actual_width": 0, "actual_height": 0}

        # 3. 너비 검증 (expected_width가 있을 경우)
        if expected_width:
            width_ratio = actual_width / expected_width
            if width_ratio < (1 - tolerance) or width_ratio > (1 + tolerance):
                return {
                    "valid": False,
                    "reason": f"너비 불일치: 예상 {expected_width}px, 실제 {actual_width}px ({width_ratio:.2f}배)",
                    "actual_width": actual_width,
                    "actual_height": actual_height
                }

        # 4. 높이 검증 (expected_height가 있을 경우)
        if expected_height:
            height_ratio = actual_height / expected_height
            if height_ratio < (1 - tolerance) or height_ratio > (1 + tolerance):
                return {
                    "valid": False,
                    "reason": f"높이 불일치: 예상 {expected_height}px, 실제 {actual_height}px ({height_ratio:.2f}배)",
                    "actual_width": actual_width,
                    "actual_height": actual_height
                }

        # 5. 최소 크기 검증
        if actual_width < 50 or actual_height < 50:
            return {
                "valid": False,
                "reason": f"이미지 크기 너무 작음: {actual_width}x{actual_height}",
                "actual_width": actual_width,
                "actual_height": actual_height
            }

        return {"valid": True, "reason": "검증 통과", "actual_width": actual_width, "actual_height": actual_height}

    except Exception as e:
        return {"valid": False, "reason": f"검증 중 오류: {str(e)}", "actual_width": 0, "actual_height": 0}


def validate_translated_chunk(
    chunk_base64: str,
    target_lang: str = None,
    expected_width: int = None,
    expected_height: int = None,
    size_tolerance: float = 0.3,
    lang_threshold: float = 0.2,
    skip_ocr: bool = False
) -> dict:
    """
    번역된 청크의 통합 검증 (크기 + 번역 언어)

    Args:
        chunk_base64: 검증할 번역된 이미지 base64
        target_lang: 번역 대상 언어 코드 (예: 'en', 'ja')
        expected_width: 예상 너비
        expected_height: 예상 높이
        size_tolerance: 크기 허용 오차 (기본 30%)
        lang_threshold: 비타겟 언어 허용 비율 (기본 20%)
        skip_ocr: OCR 검증 건너뛰기

    Returns:
        dict: {
            "valid": bool,
            "defect_type": str or None,  # "size", "translation", None
            "reason": str,
            "size_validation": dict,
            "translation_validation": dict or None,
            "can_retry": bool  # 재처리 가능 여부
        }
    """
    result = {
        "valid": True,
        "defect_type": None,
        "reason": "검증 통과",
        "size_validation": None,
        "translation_validation": None,
        "can_retry": True
    }

    # 1. 크기 검증
    size_result = validate_chunk(
        chunk_base64,
        expected_width=expected_width,
        expected_height=expected_height,
        tolerance=size_tolerance
    )
    result["size_validation"] = size_result

    if not size_result["valid"]:
        result["valid"] = False
        result["defect_type"] = "size"
        result["reason"] = size_result["reason"]

        # 크기가 너무 작으면 재처리 불가 (원본 데이터 손상)
        if size_result.get("actual_width", 0) < 50 or size_result.get("actual_height", 0) < 50:
            result["can_retry"] = False

        return result

    # 2. 번역 언어 검증 (target_lang이 있고 skip_ocr이 False인 경우)
    if target_lang and not skip_ocr:
        try:
            lang_result = validate_translation_language(
                chunk_base64,
                target_lang,
                threshold=lang_threshold
            )
            result["translation_validation"] = lang_result

            if not lang_result["valid"]:
                result["valid"] = False
                result["defect_type"] = "translation"
                result["reason"] = lang_result["reason"]
                result["can_retry"] = True  # 번역 실패는 재처리 가능
                return result

        except Exception as e:
            logger.warning(f"번역 언어 검증 스킵 (오류): {e}")
            result["translation_validation"] = {
                "valid": True,
                "reason": f"검증 오류로 통과 처리: {str(e)}",
                "has_text": False
            }

    result["reason"] = "모든 검증 통과"
    return result


def retry_chunk_realtime(chunk_base64, gemini_api_key, prompt, gemini_model="gemini-3-pro-image-preview", max_retries=2):
    """
    실시간 API로 청크 재처리

    Args:
        chunk_base64: 원본 청크 이미지 base64
        gemini_api_key: Gemini API 키
        prompt: 번역 프롬프트
        gemini_model: 사용할 모델
        max_retries: 최대 재시도 횟수

    Returns:
        dict: {
            "success": bool,
            "base64": str (번역된 이미지 또는 원본),
            "error": str (에러 메시지)
        }
    """
    import time

    BASE_DELAY = 1.0
    MAX_DELAY = 30.0

    for attempt in range(max_retries):
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"

            request_body = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": chunk_base64}}
                    ]
                }],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"]
                }
            }

            response = requests.post(
                api_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=180
            )

            if response.status_code == 429 or response.status_code >= 500:
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                logger.warning(f"[retry_chunk] {response.status_code} 발생, {delay:.1f}초 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(delay)
                continue

            response.raise_for_status()
            result = response.json()

            # 응답에서 이미지 추출
            candidates = result.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts"):
                for part in candidates[0]["content"]["parts"]:
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if inline_data and inline_data.get("data") and len(inline_data["data"]) > 1000:
                        return {"success": True, "base64": inline_data["data"], "error": None}

            # 이미지 없이 응답
            finish_reason = candidates[0].get("finishReason", "") if candidates else "NO_CANDIDATES"
            return {"success": False, "base64": chunk_base64, "error": f"이미지 응답 없음 ({finish_reason})"}

        except requests.exceptions.Timeout:
            logger.warning(f"[retry_chunk] 타임아웃 ({attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
            continue

        except Exception as e:
            logger.error(f"[retry_chunk] 오류: {str(e)}")
            return {"success": False, "base64": chunk_base64, "error": str(e)}

    return {"success": False, "base64": chunk_base64, "error": f"{max_retries}회 재시도 실패"}


def image_to_base64(image, format="JPEG", quality=95):
    """이미지를 Base64로 변환 (JPEG, PNG, WEBP 지원)

    WebP 제한: 최대 16383 픽셀 (너비/높이)
    제한 초과 시 자동으로 JPEG로 폴백
    """
    buffer = BytesIO()
    format_upper = format.upper()
    actual_format = format_upper

    # WebP 크기 제한 체크 (16383 픽셀)
    WEBP_MAX_DIMENSION = 16383
    width, height = image.size

    if format_upper == "WEBP" and (width > WEBP_MAX_DIMENSION or height > WEBP_MAX_DIMENSION):
        logger.warning(f"Image size {width}x{height} exceeds WebP limit ({WEBP_MAX_DIMENSION}px). Falling back to JPEG.")
        actual_format = "JPEG"

    if actual_format == "PNG":
        image.save(buffer, format="PNG")
    elif actual_format == "WEBP":
        # WEBP: 손실 압축, 품질 설정 가능 (0-100)
        image.save(buffer, format="WEBP", quality=quality, method=4)
    else:
        # JPEG (기본값 또는 폴백)
        image.save(buffer, format="JPEG", quality=quality)

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), actual_format


def pil_to_cv2(pil_image):
    """PIL Image를 OpenCV 형식으로 변환"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "text-render-service-v9",
        "vertex_ai_available": vertex_ai_available,
        "project_id": PROJECT_ID,
        "features": ["slice", "merge", "batch-results", "translate-chunks", "prepare-batch", "ocr-validation", "original-chunk-preservation", "retry-queue"]
    })


@app.route("/validate-translation", methods=["POST"])
def validate_translation_endpoint():
    """
    번역된 이미지의 언어 검증 테스트 엔드포인트

    Request:
        image_base64: 검증할 이미지의 base64 (필수)
        target_lang: 대상 언어 코드 (기본: 'en')
        threshold: 비타겟 언어 허용 비율 (기본: 0.2)
        expected_width: 예상 너비 (선택)
        expected_height: 예상 높이 (선택)

    Response:
        valid: 검증 통과 여부
        defect_type: 불량 유형 (size/translation/None)
        reason: 검증 결과 사유
        details: 상세 검증 결과
    """
    try:
        data = request.get_json()

        image_base64 = data.get("image_base64")
        if not image_base64:
            return jsonify({"error": "image_base64 required"}), 400

        target_lang = data.get("target_lang", "en")
        threshold = float(data.get("threshold", 0.2))
        expected_width = data.get("expected_width")
        expected_height = data.get("expected_height")

        # 통합 검증 수행
        result = validate_translated_chunk(
            chunk_base64=image_base64,
            target_lang=target_lang,
            expected_width=expected_width,
            expected_height=expected_height,
            lang_threshold=threshold,
            skip_ocr=False
        )

        return jsonify({
            "valid": result["valid"],
            "defect_type": result.get("defect_type"),
            "reason": result["reason"],
            "can_retry": result.get("can_retry", True),
            "details": {
                "size_validation": result.get("size_validation"),
                "translation_validation": result.get("translation_validation")
            }
        })

    except Exception as e:
        logger.error(f"Translation validation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ocr-detect", methods=["POST"])
def ocr_detect_endpoint():
    """
    이미지에서 텍스트 감지 (OCR) 테스트 엔드포인트

    Request:
        image_base64: 이미지의 base64 (필수)
        lang: OCR 언어 코드 (기본: 'en')

    Response:
        has_text: 텍스트 존재 여부
        total_chars: 감지된 총 문자 수
        detected_texts: 감지된 텍스트 목록
    """
    try:
        data = request.get_json()

        image_base64 = data.get("image_base64")
        if not image_base64:
            return jsonify({"error": "image_base64 required"}), 400

        lang = data.get("lang", "en")

        # OCR 수행
        result = validate_translation_language(image_base64, lang)

        return jsonify({
            "has_text": result.get("has_text", False),
            "total_chars": result.get("total_chars", 0),
            "target_lang_ratio": result.get("target_lang_ratio", 1.0),
            "detected_texts": result.get("detected_text", [])
        })

    except Exception as e:
        logger.error(f"OCR detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/cleanup-original-chunks", methods=["POST"])
def cleanup_original_chunks_endpoint():
    """
    배치 처리 완료 후 원본 청크 정리

    Request:
        batch_id: 정리할 배치 ID (필수)
        supabase_url: Supabase URL (선택, 환경변수 대체)
        supabase_key: Supabase 키 (선택, 환경변수 대체)

    Response:
        success: 성공 여부
        deleted_count: 삭제된 파일 수
    """
    try:
        data = request.get_json()

        batch_id = data.get("batch_id")
        if not batch_id:
            return jsonify({"error": "batch_id required"}), 400

        supabase_url = data.get("supabase_url")
        supabase_key = data.get("supabase_key")

        result = delete_original_chunks(
            batch_id=batch_id,
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )

        if result["success"]:
            logger.info(f"[cleanup] 원본 청크 정리 완료: batch_id={batch_id}, 삭제={result['deleted_count']}")
            return jsonify({
                "success": True,
                "batch_id": batch_id,
                "deleted_count": result["deleted_count"]
            })
        else:
            logger.warning(f"[cleanup] 원본 청크 정리 실패: {result['error']}")
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get-original-chunk", methods=["POST"])
def get_original_chunk_endpoint():
    """
    원본 청크 조회 (디버깅/테스트용)

    Request:
        batch_id: 배치 ID (필수)
        chunk_key: 청크 키 (필수, 예: "p123_i0_c0")
        supabase_url: Supabase URL (선택)
        supabase_key: Supabase 키 (선택)

    Response:
        success: 성공 여부
        base64: 원본 이미지 base64
    """
    try:
        data = request.get_json()

        batch_id = data.get("batch_id")
        chunk_key = data.get("chunk_key")

        if not batch_id or not chunk_key:
            return jsonify({"error": "batch_id and chunk_key required"}), 400

        result = get_original_chunk(
            chunk_key=chunk_key,
            batch_id=batch_id,
            supabase_url=data.get("supabase_url"),
            supabase_key=data.get("supabase_key")
        )

        if result["success"]:
            return jsonify({
                "success": True,
                "batch_id": batch_id,
                "chunk_key": chunk_key,
                "base64": result["base64"]
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 404

    except Exception as e:
        logger.error(f"Get original chunk error: {e}")
        return jsonify({"error": str(e)}), 500


# ===== 재처리 대기열 엔드포인트 =====

@app.route("/retry-queue/add", methods=["POST"])
def add_to_retry_queue():
    """
    실패한 청크를 재처리 대기열에 추가

    Request:
        chunk_info: 청크 정보 {batch_id, chunk_key, product_id, ...}
        config: 처리 설정 {targetLangCode, prompt, geminiApiKey, ...}
        error_reason: 실패 사유

    Response:
        success: 성공 여부
        queue_id: 대기열 ID
    """
    try:
        data = request.get_json()

        chunk_info = data.get("chunk_info", {})
        config = data.get("config", {})
        error_reason = data.get("error_reason", "Unknown error")

        if not chunk_info.get("chunk_key"):
            return jsonify({"error": "chunk_info.chunk_key required"}), 400

        result = enqueue_failed_chunk(
            chunk_info=chunk_info,
            config=config,
            error_reason=error_reason,
            supabase_url=config.get("supabaseUrl"),
            supabase_key=config.get("supabaseKey")
        )

        if result["success"]:
            return jsonify({
                "success": True,
                "queue_id": result["queue_id"],
                "chunk_key": chunk_info.get("chunk_key")
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500

    except Exception as e:
        logger.error(f"Add to retry queue error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retry-queue/list", methods=["GET", "POST"])
def list_retry_queue():
    """
    재처리 대기열 조회

    Request (POST) or Query Params (GET):
        batch_id: 특정 배치만 조회 (선택)
        status: 상태 필터 (선택, 기본: pending)
        limit: 최대 조회 수 (선택, 기본: 50)

    Response:
        success: 성공 여부
        items: 대기열 항목 리스트
        count: 항목 수
    """
    try:
        if request.method == "POST":
            data = request.get_json() or {}
        else:
            data = {}

        batch_id = data.get("batch_id") or request.args.get("batch_id")
        limit = int(data.get("limit") or request.args.get("limit", 50))

        result = get_pending_queue_items(
            batch_id=batch_id,
            limit=limit,
            supabase_url=data.get("supabase_url"),
            supabase_key=data.get("supabase_key")
        )

        if result["success"]:
            return jsonify({
                "success": True,
                "items": result["items"],
                "count": len(result["items"])
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500

    except Exception as e:
        logger.error(f"List retry queue error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retry-queue/process", methods=["POST"])
def process_retry_queue():
    """
    재처리 대기열 항목들 처리

    Request:
        batch_id: 특정 배치만 처리 (선택)
        limit: 최대 처리 수 (선택, 기본: 10)
        supabase_url: Supabase URL (선택)
        supabase_key: Supabase 키 (선택)

    Response:
        success: 성공 여부
        processed: 처리된 항목 수
        succeeded: 성공한 항목 수
        failed: 실패한 항목 수
        images_completed: 완료된 이미지 목록
    """
    try:
        data = request.get_json() or {}

        batch_id = data.get("batch_id")
        limit = int(data.get("limit", 10))
        supabase_url = data.get("supabase_url") or SUPABASE_URL
        supabase_key = data.get("supabase_key") or SUPABASE_SERVICE_KEY

        # 대기 중인 항목 조회
        queue_result = get_pending_queue_items(
            batch_id=batch_id,
            limit=limit,
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )

        if not queue_result["success"]:
            return jsonify({
                "success": False,
                "error": queue_result["error"]
            }), 500

        items = queue_result["items"]
        if not items:
            return jsonify({
                "success": True,
                "message": "처리할 항목 없음",
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "images_completed": []
            })

        # 각 항목 처리
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "images_completed": []
        }

        for item in items:
            result = process_single_retry_item(
                item=item,
                supabase_url=supabase_url,
                supabase_key=supabase_key
            )

            results["processed"] += 1

            if result["success"]:
                results["succeeded"] += 1
                if result.get("image_complete"):
                    results["images_completed"].append({
                        "batch_id": item.get("batch_id"),
                        "product_id": item.get("product_id"),
                        "image_index": item.get("image_index"),
                        "completion_info": result.get("completion_info")
                    })
            else:
                results["failed"] += 1

        logger.info(f"[대기열 처리 완료] 처리={results['processed']}, 성공={results['succeeded']}, 실패={results['failed']}")

        return jsonify({
            "success": True,
            **results
        })

    except Exception as e:
        logger.error(f"Process retry queue error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retry-queue/check-and-merge", methods=["POST"])
def check_and_merge_image():
    """
    이미지 완료 여부 확인 및 병합 실행

    Request:
        batch_id: 배치 ID (필수)
        product_id: 상품 ID (필수)
        image_index: 이미지 인덱스 (필수)
        total_chunks: 총 청크 수 (필수)
        config: 설정 {supabaseUrl, supabaseKey, storageBucket, outputFormat, ...}

    Response:
        complete: 완료 여부
        merged: 병합 실행 여부
        uploaded_url: 업로드된 URL (병합 시)
    """
    try:
        data = request.get_json()

        batch_id = data.get("batch_id")
        product_id = data.get("product_id")
        image_index = data.get("image_index")
        total_chunks = data.get("total_chunks")
        config = data.get("config", {})

        if not all([batch_id, product_id is not None, image_index is not None, total_chunks]):
            return jsonify({"error": "batch_id, product_id, image_index, total_chunks required"}), 400

        supabase_url = config.get("supabaseUrl") or SUPABASE_URL
        supabase_key = config.get("supabaseKey") or SUPABASE_SERVICE_KEY

        # 완료 여부 확인
        completion = check_image_completion(
            batch_id=batch_id,
            product_id=product_id,
            image_index=image_index,
            total_chunks=total_chunks,
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )

        if not completion["complete"]:
            return jsonify({
                "complete": False,
                "merged": False,
                "completed_count": completion["completed_count"],
                "total_chunks": total_chunks
            })

        # 청크들을 순서대로 정렬하여 병합
        chunks = sorted(completion["chunks"], key=lambda x: x.get("chunk_index", 0))

        # 병합을 위한 청크 데이터 준비
        chunk_data_list = []
        for chunk in chunks:
            chunk_data_list.append({
                "base64": chunk.get("translated_base64"),
                "index": chunk.get("chunk_index"),
                "height": chunk.get("chunk_height", 2000)
            })

        # 병합 실행
        original_width = chunks[0].get("chunk_width", 1000) if chunks else 1000

        merged_result = merge_images_internal(
            chunks=chunk_data_list,
            original_width=original_width,
            output_format=config.get("outputFormat", "WEBP"),
            output_quality=config.get("outputQuality", 100)
        )

        if not merged_result["success"]:
            return jsonify({
                "complete": True,
                "merged": False,
                "error": merged_result.get("error", "병합 실패")
            }), 500

        # Supabase Storage에 업로드
        storage_bucket = config.get("storageBucket", "translated-images")
        product_code = chunks[0].get("product_code", f"product_{product_id}")
        output_extension = config.get("outputExtension", ".webp")
        target_lang = config.get("targetLangCode", "en")

        storage_path = f"{product_code}/{target_lang}/image_{image_index}{output_extension}"

        upload_result = upload_to_supabase_storage(
            image_base64=merged_result["merged_base64"],
            storage_path=storage_path,
            storage_bucket=storage_bucket,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            content_type=f"image/{config.get('outputFormat', 'webp').lower()}"
        )

        if upload_result["success"]:
            logger.info(f"[병합 완료] product_id={product_id}, image_index={image_index}, url={upload_result['url']}")
            return jsonify({
                "complete": True,
                "merged": True,
                "uploaded_url": upload_result["url"],
                "storage_path": storage_path
            })
        else:
            return jsonify({
                "complete": True,
                "merged": True,
                "uploaded": False,
                "error": upload_result.get("error", "업로드 실패"),
                "merged_base64": merged_result["merged_base64"][:100] + "..."  # 일부만 반환
            }), 500

    except Exception as e:
        logger.error(f"Check and merge error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def merge_images_internal(chunks: list, original_width: int, output_format: str = "WEBP", output_quality: int = 100) -> dict:
    """
    청크 이미지들을 내부적으로 병합 (엔드포인트 호출 없이)

    Args:
        chunks: 청크 리스트 [{"base64": ..., "index": ..., "height": ...}, ...]
        original_width: 원본 이미지 너비
        output_format: 출력 포맷
        output_quality: 출력 품질

    Returns:
        dict: {"success": bool, "merged_base64": str, "error": str}
    """
    try:
        if not chunks:
            return {"success": False, "merged_base64": "", "error": "청크 없음"}

        # 청크 순서대로 정렬
        sorted_chunks = sorted(chunks, key=lambda x: x.get("index", 0))

        # 각 청크 이미지 디코딩
        images = []
        for chunk in sorted_chunks:
            chunk_base64 = chunk.get("base64")
            if not chunk_base64:
                continue
            try:
                img_bytes = base64.b64decode(chunk_base64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"청크 디코딩 실패: {e}")
                continue

        if not images:
            return {"success": False, "merged_base64": "", "error": "유효한 청크 없음"}

        # 총 높이 계산
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)

        # 새 이미지 생성
        merged = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        # 청크들을 순서대로 붙이기
        current_y = 0
        for img in images:
            merged.paste(img, (0, current_y))
            current_y += img.height

        # base64로 인코딩
        buffer = BytesIO()
        pil_format = output_format.upper()
        if pil_format == "JPG":
            pil_format = "JPEG"

        if pil_format == "WEBP":
            merged.save(buffer, format="WEBP", quality=output_quality, method=4)
        elif pil_format == "JPEG":
            merged.save(buffer, format="JPEG", quality=output_quality, optimize=True)
        else:
            merged.save(buffer, format=pil_format, quality=output_quality)

        merged_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "success": True,
            "merged_base64": merged_base64,
            "width": max_width,
            "height": total_height,
            "error": ""
        }

    except Exception as e:
        return {"success": False, "merged_base64": "", "error": str(e)}


def upload_to_supabase_storage(
    image_base64: str,
    storage_path: str,
    storage_bucket: str,
    supabase_url: str,
    supabase_key: str,
    content_type: str = "image/webp"
) -> dict:
    """
    Supabase Storage에 이미지 업로드

    Args:
        image_base64: 이미지 base64
        storage_path: 저장 경로
        storage_bucket: 버킷명
        supabase_url: Supabase URL
        supabase_key: Supabase 키
        content_type: 컨텐츠 타입

    Returns:
        dict: {"success": bool, "url": str, "error": str}
    """
    try:
        image_bytes = base64.b64decode(image_base64)

        upload_url = f"{supabase_url}/storage/v1/object/{storage_bucket}/{storage_path}"

        # 먼저 업로드 시도
        response = requests.post(
            upload_url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": content_type
            },
            data=image_bytes,
            timeout=60
        )

        # 이미 존재하면 업데이트
        if response.status_code == 400 and "already exists" in response.text.lower():
            response = requests.put(
                upload_url,
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": content_type
                },
                data=image_bytes,
                timeout=60
            )

        if response.status_code in [200, 201]:
            public_url = f"{supabase_url}/storage/v1/object/public/{storage_bucket}/{storage_path}"
            return {"success": True, "url": public_url, "error": ""}
        else:
            return {
                "success": False,
                "url": "",
                "error": f"업로드 실패: {response.status_code} - {response.text[:200]}"
            }

    except Exception as e:
        return {"success": False, "url": "", "error": str(e)}


# ===== 이미지 슬라이스/병합 기능 =====

def find_smart_cut_point(image, target_y, search_range=200):
    """
    텍스트가 없는 최적의 자르기 위치 찾기 (최적화 버전)
    - target_y 근처에서 색상 분산이 가장 낮은 가로 라인을 찾음
    - 색상 분산이 낮음 = 단색 배경 = 텍스트 없음
    - NumPy 벡터 연산 + 서브샘플링으로 약 5배 빠름
    """
    cv_image = pil_to_cv2(image)
    height = cv_image.shape[0]
    width = cv_image.shape[1]

    min_y = max(0, target_y - search_range)
    max_y = min(height - 1, target_y + search_range)

    # 1단계: 서브샘플링으로 빠르게 후보 영역 찾기 (5픽셀 간격)
    step = 5
    y_candidates = np.arange(min_y, max_y, step)

    if len(y_candidates) == 0:
        return target_y

    # 이미지의 일부만 샘플링 (너비의 20%만 사용, 균등 분포)
    sample_cols = np.linspace(0, width - 1, min(width, max(50, width // 5)), dtype=int)

    # 각 라인의 분산을 벡터 연산으로 계산
    region = cv_image[min_y:max_y:step, sample_cols, :]  # (num_lines, num_samples, 3)
    variances = np.var(region, axis=(1, 2))  # 각 라인의 분산

    # 가장 낮은 분산을 가진 후보 찾기
    best_idx = np.argmin(variances)
    coarse_best_y = y_candidates[best_idx]

    # 2단계: 후보 근처에서 정밀 검색 (1픽셀 간격)
    fine_min_y = max(min_y, coarse_best_y - step)
    fine_max_y = min(max_y, coarse_best_y + step)

    best_y = coarse_best_y
    min_variance = variances[best_idx]

    for y in range(fine_min_y, fine_max_y):
        line = cv_image[y, sample_cols, :]
        variance = np.var(line)
        if variance < min_variance:
            min_variance = variance
            best_y = y

    logger.info(f"Smart cut: target={target_y}, best={best_y}, variance={min_variance:.2f}")
    return int(best_y)  # NumPy int64 -> Python int 변환


def slice_image(image, chunk_height=3000, use_smart_cut=True, overlap=0):
    """
    긴 이미지를 청크로 분할

    Args:
        image: PIL Image
        chunk_height: 기본 청크 높이 (픽셀)
        use_smart_cut: 텍스트 없는 위치에서 자르기
        overlap: 청크 간 오버랩 (픽셀)

    Returns:
        list of dict: [{index, y_start, y_end, height, base64}, ...]
    """
    width, height = image.size
    chunks = []

    y = 0
    index = 0

    while y < height:
        # 남은 높이 계산
        remaining = height - y

        if remaining <= chunk_height:
            # 마지막 청크
            chunk_end = height
        else:
            # 기본 자르기 위치
            target_cut = y + chunk_height

            if use_smart_cut and remaining > chunk_height:
                # 스마트 컷: 텍스트 없는 위치 찾기
                chunk_end = find_smart_cut_point(image, target_cut, search_range=200)
            else:
                chunk_end = target_cut

        # 청크 추출
        chunk_image = image.crop((0, y, width, chunk_end))
        chunk_base64, _ = image_to_base64(chunk_image)

        chunks.append({
            "index": index,
            "y_start": y,
            "y_end": chunk_end,
            "height": chunk_end - y,
            "width": width,
            "base64": chunk_base64
        })

        logger.info(f"Chunk {index}: y={y}-{chunk_end}, height={chunk_end - y}")

        # 다음 청크 시작점 (오버랩 적용)
        y = chunk_end - overlap
        index += 1

    return chunks


def blend_regions_numpy(prev_img, curr_img, blend_height, overlap, target_width):
    """
    NumPy 벡터 연산을 사용한 고속 이미지 블렌딩
    기존 픽셀별 반복문 대비 약 50배 빠름
    """
    # PIL to NumPy
    prev_array = np.array(prev_img)
    curr_array = np.array(curr_img)

    blend_region_height = min(blend_height, overlap)

    # 이전 이미지의 블렌딩 영역 (하단)
    prev_start = prev_img.size[1] - overlap
    prev_region = prev_array[prev_start:prev_start + blend_region_height, :, :].astype(np.float32)

    # 현재 이미지의 블렌딩 영역 (상단)
    curr_region = curr_array[:blend_region_height, :, :].astype(np.float32)

    # 알파 그라데이션 생성 (0 → 1)
    alpha = np.linspace(0, 1, blend_region_height).reshape(-1, 1, 1)

    # 벡터 연산으로 블렌딩
    blended = (prev_region * (1 - alpha) + curr_region * alpha).astype(np.uint8)

    return Image.fromarray(blended)


def merge_images(chunks, overlap=0, blend_height=50, target_width=None, target_heights=None):
    """
    청크들을 하나의 이미지로 병합 (NumPy 최적화 버전)

    Gemini가 반환하는 이미지 크기가 원본과 다를 수 있으므로,
    필요시 리사이즈하여 병합합니다.

    Args:
        chunks: list of dict with 'base64', 'index', 'height' (원본 height)
        overlap: 오버랩 픽셀 수
        blend_height: 블렌딩 영역 높이
        target_width: 목표 너비 (없으면 첫 번째 이미지 기준)
        target_heights: 각 청크의 목표 높이 리스트 (없으면 원본 height 사용)

    Returns:
        PIL Image
    """
    if not chunks:
        raise ValueError("No chunks to merge")

    # index 순서로 정렬
    sorted_chunks = sorted(chunks, key=lambda x: x.get("index", 0))

    # 모든 청크 이미지를 먼저 로드하고 필요시 리사이즈
    chunk_images = []
    first_width = None

    for i, chunk_data in enumerate(sorted_chunks):
        chunk_bytes = base64.b64decode(chunk_data["base64"])
        chunk_image = Image.open(BytesIO(chunk_bytes)).convert("RGB")

        original_height = chunk_data.get("height", chunk_image.size[1])

        if first_width is None:
            first_width = chunk_image.size[0]
            if target_width is None:
                target_width = first_width

        # Gemini가 반환한 이미지 크기가 다르면 원본 크기로 리사이즈
        actual_width, actual_height = chunk_image.size

        if actual_width != target_width or actual_height != original_height:
            logger.info(f"Chunk {i}: resizing from {actual_width}x{actual_height} to {target_width}x{original_height}")
            chunk_image = chunk_image.resize((target_width, original_height), Image.Resampling.LANCZOS)

        chunk_images.append({
            "image": chunk_image,
            "height": chunk_image.size[1],
            "original_height": original_height
        })

    # 전체 높이 계산 (실제 이미지 높이 기준)
    total_height = sum(c["height"] for c in chunk_images)
    if overlap > 0:
        total_height -= overlap * (len(chunk_images) - 1)

    # 결과 이미지 생성
    result = Image.new("RGB", (target_width, total_height))
    logger.info(f"Creating merged image: {target_width}x{total_height}")

    current_y = 0
    prev_chunk_img = None

    for i, chunk_data in enumerate(chunk_images):
        chunk_image = chunk_data["image"]
        chunk_height = chunk_data["height"]

        if i == 0:
            # 첫 번째 청크는 그대로 붙임
            result.paste(chunk_image, (0, current_y))
            current_y += chunk_height - overlap
        else:
            if overlap > 0 and blend_height > 0 and prev_chunk_img is not None:
                # NumPy 벡터 연산으로 고속 블렌딩
                blend_region_height = min(blend_height, overlap)
                blended_region = blend_regions_numpy(
                    prev_chunk_img, chunk_image, blend_height, overlap, target_width
                )
                result.paste(blended_region, (0, current_y))

                # 블렌딩 영역 이후 나머지 붙이기
                remaining = chunk_image.crop((0, blend_region_height, target_width, chunk_height))
                result.paste(remaining, (0, current_y + blend_region_height))
            else:
                if overlap > 0:
                    cropped = chunk_image.crop((0, overlap, target_width, chunk_height))
                    result.paste(cropped, (0, current_y))
                else:
                    result.paste(chunk_image, (0, current_y))

            current_y += chunk_height - overlap

        prev_chunk_img = chunk_image

    logger.info(f"Merged {len(chunk_images)} chunks into {target_width}x{total_height} image (current_y={current_y})")
    return result


@app.route("/slice-image", methods=["POST"])
def slice_image_endpoint():
    """
    긴 이미지를 청크로 분할

    Request:
        image_base64: Base64 인코딩된 이미지
        chunk_height: 청크 높이 (기본 3000)
        use_smart_cut: 스마트 컷 사용 여부 (기본 True)
        overlap: 오버랩 픽셀 (기본 0)

    Response:
        chunks: [{index, y_start, y_end, height, width, base64}, ...]
        total_chunks: 총 청크 수
        original_size: {width, height}
    """
    try:
        data = request.get_json()
        image_base64_input = data.get("image_base64")
        chunk_height = data.get("chunk_height", 3000)
        use_smart_cut = data.get("use_smart_cut", True)
        overlap = data.get("overlap", 0)

        if not image_base64_input:
            return jsonify({"error": "image_base64 required"}), 400

        # 이미지 로드
        image_data = base64.b64decode(image_base64_input)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        original_width, original_height = image.size

        logger.info(f"Slicing image: {original_width}x{original_height}, chunk_height={chunk_height}")

        # 이미지가 chunk_height보다 작으면 슬라이스 불필요
        if original_height <= chunk_height:
            return jsonify({
                "success": True,
                "chunks": [{
                    "index": 0,
                    "y_start": 0,
                    "y_end": original_height,
                    "height": original_height,
                    "width": original_width,
                    "base64": image_base64_input
                }],
                "total_chunks": 1,
                "original_size": {"width": original_width, "height": original_height},
                "needs_merge": False
            })

        # 슬라이스 실행
        chunks = slice_image(image, chunk_height, use_smart_cut, overlap)

        return jsonify({
            "success": True,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "original_size": {"width": original_width, "height": original_height},
            "needs_merge": True,
            "overlap": overlap
        })

    except Exception as e:
        logger.error(f"Error in slice_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/merge-images", methods=["POST"])
def merge_images_endpoint():
    """
    청크들을 하나의 이미지로 병합

    Request:
        chunks: [{index, base64, height}, ...]
        overlap: 오버랩 픽셀 (기본 0)
        blend_height: 블렌딩 영역 높이 (기본 50)
        target_width: 원본 이미지 너비 (필수 권장)
        target_heights: 각 청크의 원본 높이 리스트
        output_format: 출력 포맷 - JPEG, PNG, WEBP (기본 JPEG)
        quality: 이미지 품질 0-100 (기본 95)

        # Supabase 직접 업로드 옵션 (n8n 메모리 문제 해결용)
        upload_to_supabase: true면 직접 업로드하고 URL 반환
        supabase_url: Supabase URL
        supabase_key: Supabase API 키
        storage_bucket: 스토리지 버킷명
        file_name: 업로드할 파일 경로

    Response:
        (upload_to_supabase=false): image_base64, size, format
        (upload_to_supabase=true): uploaded_url, size, format
    """
    try:
        data = request.get_json()
        chunks = data.get("chunks", [])
        overlap = data.get("overlap", 0)
        blend_height = data.get("blend_height", 50)
        target_width = data.get("target_width")  # 원본 너비
        target_heights = data.get("target_heights")  # 각 청크의 원본 높이
        output_format = data.get("output_format", "JPEG").upper()  # JPEG, PNG, WEBP
        quality = data.get("quality", 95)  # 이미지 품질

        # Supabase 직접 업로드 옵션
        upload_to_supabase = data.get("upload_to_supabase", False)
        supabase_url = data.get("supabase_url")
        supabase_key = data.get("supabase_key")
        storage_bucket = data.get("storage_bucket")
        file_name = data.get("file_name")

        if not chunks:
            return jsonify({"error": "chunks required"}), 400

        # 지원 포맷 검증
        if output_format not in ["JPEG", "PNG", "WEBP", "JPG"]:
            output_format = "JPEG"

        # JPG를 JPEG로 통일
        if output_format == "JPG":
            output_format = "JPEG"

        logger.info(f"Merging {len(chunks)} chunks, overlap={overlap}, target_width={target_width}, format={output_format}, upload_to_supabase={upload_to_supabase}")

        # 병합 실행 (원본 크기로 강제 리사이즈)
        merged_image = merge_images(chunks, overlap, blend_height, target_width, target_heights)
        width, height = merged_image.size

        # WebP 크기 제한 체크 (16383 픽셀)
        WEBP_MAX_DIMENSION = 16383

        # WebP 요청이고 크기 제한 초과 시 분할 저장
        if output_format == "WEBP" and height > WEBP_MAX_DIMENSION:
            logger.info(f"Image height {height}px exceeds WebP limit ({WEBP_MAX_DIMENSION}px). Splitting into multiple WebP files.")

            split_images = []
            y = 0
            part_num = 1

            while y < height:
                # 각 파트의 높이 (마지막 파트는 남은 높이)
                chunk_height = min(WEBP_MAX_DIMENSION, height - y)

                # 이미지 크롭
                chunk = merged_image.crop((0, y, width, y + chunk_height))

                # WebP로 인코딩
                buffer = BytesIO()
                chunk.save(buffer, format="WEBP", quality=quality, method=4)
                chunk_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                split_images.append({
                    "base64": chunk_base64,
                    "part": part_num,
                    "height": chunk_height,
                    "y_start": y,
                    "y_end": y + chunk_height
                })

                logger.info(f"  Part {part_num}: y={y}-{y + chunk_height}, height={chunk_height}px")

                y += chunk_height
                part_num += 1

            logger.info(f"Split into {len(split_images)} parts")

            # 분할된 이미지도 Supabase 직접 업로드 지원
            if upload_to_supabase and supabase_url and supabase_key and storage_bucket and file_name:
                logger.info(f"Uploading {len(split_images)} split images to Supabase")

                uploaded_urls = []
                for img_part in split_images:
                    # 파일명에 파트 번호 추가 (확장자 앞에)
                    base_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                    ext = file_name.rsplit('.', 1)[1] if '.' in file_name else 'webp'
                    part_file_name = f"{base_name}_{str(img_part['part']).zfill(2)}.{ext}"

                    upload_url = f"{supabase_url}/storage/v1/object/{storage_bucket}/{part_file_name}"

                    try:
                        upload_response = requests.post(
                            upload_url,
                            headers={
                                "Authorization": f"Bearer {supabase_key}",
                                "Content-Type": "image/webp",
                                "x-upsert": "true"
                            },
                            data=base64.b64decode(img_part["base64"]),
                            timeout=120
                        )

                        if upload_response.status_code in [200, 201]:
                            public_url = f"{supabase_url}/storage/v1/object/public/{storage_bucket}/{part_file_name}"
                            uploaded_urls.append({
                                "part": img_part["part"],
                                "url": public_url,
                                "height": img_part["height"]
                            })
                            logger.info(f"  Part {img_part['part']} uploaded: {public_url}")
                        else:
                            logger.error(f"  Part {img_part['part']} upload failed: {upload_response.status_code}")

                    except Exception as part_err:
                        logger.error(f"  Part {img_part['part']} upload error: {part_err}")

                return jsonify({
                    "success": True,
                    "split": True,
                    "uploaded": True,
                    "uploaded_urls": uploaded_urls,
                    "total_parts": len(split_images),
                    "size": {"width": width, "height": height},
                    "format": "WEBP",
                    "mime_type": "image/webp",
                    "extension": ".webp"
                })

            return jsonify({
                "success": True,
                "split": True,
                "images": split_images,
                "total_parts": len(split_images),
                "size": {"width": width, "height": height},
                "format": "WEBP",
                "mime_type": "image/webp",
                "extension": ".webp"
            })

        # 단일 이미지 (분할 불필요)
        merged_base64, actual_format = image_to_base64(merged_image, format=output_format, quality=quality)

        # 실제 사용된 포맷에 따른 MIME 타입과 확장자
        format_info = {
            "JPEG": {"mime_type": "image/jpeg", "extension": ".jpg"},
            "PNG": {"mime_type": "image/png", "extension": ".png"},
            "WEBP": {"mime_type": "image/webp", "extension": ".webp"}
        }
        info = format_info.get(actual_format, format_info["JPEG"])

        # Supabase 직접 업로드 모드
        if upload_to_supabase and supabase_url and supabase_key and storage_bucket and file_name:
            logger.info(f"Uploading directly to Supabase: {file_name}")

            upload_url = f"{supabase_url}/storage/v1/object/{storage_bucket}/{file_name}"

            try:
                upload_response = requests.post(
                    upload_url,
                    headers={
                        "Authorization": f"Bearer {supabase_key}",
                        "Content-Type": info["mime_type"],
                        "x-upsert": "true"
                    },
                    data=base64.b64decode(merged_base64),
                    timeout=120
                )

                if upload_response.status_code in [200, 201]:
                    public_url = f"{supabase_url}/storage/v1/object/public/{storage_bucket}/{file_name}"
                    logger.info(f"Upload successful: {public_url}")

                    return jsonify({
                        "success": True,
                        "split": False,
                        "uploaded": True,
                        "uploaded_url": public_url,
                        "file_name": file_name,
                        "size": {
                            "width": merged_image.size[0],
                            "height": merged_image.size[1]
                        },
                        "format": actual_format,
                        "mime_type": info["mime_type"],
                        "extension": info["extension"]
                    })
                else:
                    logger.error(f"Upload failed: {upload_response.status_code} - {upload_response.text[:200]}")
                    return jsonify({
                        "success": False,
                        "error": f"Upload failed: HTTP {upload_response.status_code}",
                        "details": upload_response.text[:500]
                    }), 500

            except Exception as upload_err:
                logger.error(f"Upload error: {upload_err}")
                return jsonify({
                    "success": False,
                    "error": f"Upload error: {str(upload_err)}"
                }), 500

        # 기존 방식: base64 반환
        return jsonify({
            "success": True,
            "split": False,
            "image_base64": merged_base64,
            "size": {
                "width": merged_image.size[0],
                "height": merged_image.size[1]
            },
            "format": actual_format,
            "mime_type": info["mime_type"],
            "extension": info["extension"],
            "requested_format": output_format,
            "fallback_used": actual_format != output_format
        })

    except Exception as e:
        logger.error(f"Error in merge_images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ===== Gemini 배치 결과 처리 =====

@app.route("/process-batch-results", methods=["POST"])
def process_batch_results():
    """
    Gemini 배치 API 결과 파일을 처리하여 이미지 병합 및 업로드

    n8n 클라우드의 메모리 제한을 피하기 위해 이 서비스에서 직접 처리

    Request:
        gemini_file_name: Gemini 결과 파일명 (예: "files/batch-xxx")
        gemini_api_key: Gemini API 키
        chunk_metadata: 청크 메타데이터 리스트 [{key, productId, imageIndex, chunkIndex, ...}, ...]
        product_map: 상품별 정보 {productId: {productCode, imageCount}, ...}
        config: 설정 {supabaseUrl, supabaseKey, storageBucket, tableName, targetLangCode}

    Response:
        success: 성공 여부
        processed_images: 처리된 이미지 리스트
        uploaded_urls: 업로드된 URL 리스트
    """
    try:
        data = request.get_json()

        gemini_file_name = data.get("gemini_file_name")
        gemini_api_key = data.get("gemini_api_key")
        chunk_metadata = data.get("chunk_metadata", [])
        product_map = data.get("product_map", {})
        config = data.get("config", {})

        if not gemini_file_name or not gemini_api_key:
            return jsonify({"error": "gemini_file_name and gemini_api_key required"}), 400

        logger.info(f"Processing batch results: {gemini_file_name}")
        logger.info(f"Chunk metadata count: {len(chunk_metadata)}")

        # 1. Gemini 결과 파일 다운로드
        download_url = f"https://generativelanguage.googleapis.com/download/v1beta/{gemini_file_name}:download?alt=media"

        logger.info(f"Downloading from: {download_url}")

        download_response = requests.get(
            download_url,
            headers={"x-goog-api-key": gemini_api_key},
            timeout=300
        )

        if download_response.status_code != 200:
            return jsonify({
                "error": f"Failed to download results file: {download_response.status_code}",
                "details": download_response.text[:500]
            }), 400

        jsonl_content = download_response.text
        logger.info(f"Downloaded {len(jsonl_content)} bytes")

        # 2. JSONL 파싱 및 이미지별 그룹핑
        lines = jsonl_content.strip().split('\n')
        logger.info(f"Parsing {len(lines)} JSONL lines")

        # custom_id로 청크 메타데이터 인덱싱
        meta_by_key = {m["key"]: m for m in chunk_metadata}

        # 이미지별로 청크 그룹핑
        # 키: "productId_imageIndex"
        image_chunks = {}

        # 검증 통계
        validation_stats = {
            "validation_passed": 0,
            "validation_failed": 0,
            "size_defects": 0,        # 크기 불량
            "translation_defects": 0,  # 번역 불량 (OCR 검증)
            "retry_success": 0,
            "retry_failed": 0,
            "unretryable": 0,         # 재처리 불가
            "original_chunk_used": 0, # 원본 청크 사용 횟수
            "queued_for_retry": 0     # 재처리 대기열에 추가된 수
        }

        for line_num, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                response_data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON parse error - {e}")
                continue

            # Gemini Batch API uses "key" field (not "custom_id")
            custom_id = response_data.get("key") or response_data.get("custom_id") or response_data.get("customId")
            if not custom_id:
                logger.warning(f"Line {line_num}: No key/custom_id found")
                continue

            meta = meta_by_key.get(custom_id)
            if not meta:
                logger.warning(f"Line {line_num}: No metadata for {custom_id}")
                continue

            # 이미지 데이터 추출
            translated_base64 = None
            candidates = response_data.get("response", {}).get("candidates", [])

            if candidates and candidates[0].get("content", {}).get("parts"):
                for part in candidates[0]["content"]["parts"]:
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if inline_data and inline_data.get("data") and len(inline_data["data"]) > 1000:
                        translated_base64 = inline_data["data"]
                        break

            if not translated_base64:
                logger.warning(f"Line {line_num}: No image data for {custom_id}")
                continue

            # ===== 청크 검증 (크기 + OCR 번역 언어) 및 자동 재시도 =====
            expected_width = meta.get("chunkWidth")
            expected_height = meta.get("chunkHeight")

            # config에서 대상 언어 가져오기
            target_lang = config.get("targetLangCode", "en")

            # OCR 검증 활성화 여부 (기본: 활성화)
            enable_ocr_validation = config.get("enableOcrValidation", True)

            # 통합 검증 (크기 + 번역 언어)
            validation = validate_translated_chunk(
                chunk_base64=translated_base64,
                target_lang=target_lang,
                expected_width=expected_width,
                expected_height=expected_height,
                size_tolerance=0.3,  # 크기 30% 오차 허용
                lang_threshold=0.2,  # 비타겟 언어 20% 이상이면 불량
                skip_ocr=not enable_ocr_validation
            )

            if not validation["valid"]:
                defect_type = validation.get("defect_type", "unknown")
                logger.warning(f"[검증 실패] {custom_id}: {validation['reason']} (유형: {defect_type})")

                # 불량 유형별 통계 업데이트
                if defect_type == "size":
                    validation_stats["size_defects"] = validation_stats.get("size_defects", 0) + 1
                elif defect_type == "translation":
                    validation_stats["translation_defects"] = validation_stats.get("translation_defects", 0) + 1

                # 재처리 가능한 경우에만 재시도
                if validation.get("can_retry", True):
                    prompt = config.get("prompt", "Translate the text in this image to English.")
                    gemini_model = config.get("geminiModel", "gemini-3-pro-image-preview")

                    # 원본 청크 가져오기 시도
                    original_chunk_base64 = None
                    batch_id = meta.get("batchId")
                    original_chunk_path = meta.get("originalChunkPath")

                    if batch_id and original_chunk_path:
                        logger.info(f"[원본 청크 조회] {custom_id}: batch_id={batch_id}")
                        original_result = get_original_chunk(
                            chunk_key=custom_id,
                            batch_id=batch_id,
                            supabase_url=config.get("supabaseUrl") or SUPABASE_URL,
                            supabase_key=config.get("supabaseKey") or SUPABASE_SERVICE_KEY
                        )
                        if original_result["success"]:
                            original_chunk_base64 = original_result["base64"]
                            logger.info(f"[원본 청크 조회 성공] {custom_id}")
                        else:
                            logger.warning(f"[원본 청크 조회 실패] {custom_id}: {original_result['error']}")

                    # 원본 청크가 있으면 원본으로, 없으면 번역 실패본으로 재시도
                    retry_chunk = original_chunk_base64 if original_chunk_base64 else translated_base64
                    retry_source = "원본" if original_chunk_base64 else "번역본"

                    if original_chunk_base64:
                        validation_stats["original_chunk_used"] = validation_stats.get("original_chunk_used", 0) + 1

                    retry_result = retry_chunk_realtime(
                        chunk_base64=retry_chunk,
                        gemini_api_key=gemini_api_key,
                        prompt=prompt,
                        gemini_model=gemini_model,
                        max_retries=2
                    )

                    if retry_result["success"]:
                        # 재시도 결과 검증 (OCR 포함)
                        retry_validation = validate_translated_chunk(
                            chunk_base64=retry_result["base64"],
                            target_lang=target_lang,
                            expected_width=expected_width,
                            expected_height=expected_height,
                            size_tolerance=0.3,
                            lang_threshold=0.2,
                            skip_ocr=not enable_ocr_validation
                        )

                        if retry_validation["valid"]:
                            translated_base64 = retry_result["base64"]
                            logger.info(f"[재시도 성공] {custom_id}: {retry_source}으로 검증 통과")
                            validation_stats["retry_success"] = validation_stats.get("retry_success", 0) + 1
                        else:
                            logger.warning(f"[재시도 실패] {custom_id}: {retry_source}으로 재시도 후에도 검증 실패 - {retry_validation['reason']}")
                            validation_stats["retry_failed"] = validation_stats.get("retry_failed", 0) + 1

                            # 재처리 대기열에 추가
                            chunk_info = {
                                "batch_id": meta.get("batchId"),
                                "chunk_key": custom_id,
                                "product_id": meta.get("productId"),
                                "product_code": meta.get("productCode"),
                                "image_index": meta.get("imageIndex"),
                                "chunk_index": meta.get("chunkIndex"),
                                "total_chunks": meta.get("totalChunks"),
                                "chunk_width": expected_width,
                                "chunk_height": expected_height,
                                "original_chunk_path": meta.get("originalChunkPath")
                            }
                            queue_result = enqueue_failed_chunk(
                                chunk_info=chunk_info,
                                config=config,
                                error_reason=retry_validation['reason'],
                                supabase_url=config.get("supabaseUrl"),
                                supabase_key=config.get("supabaseKey")
                            )
                            if queue_result["success"]:
                                validation_stats["queued_for_retry"] = validation_stats.get("queued_for_retry", 0) + 1
                                logger.info(f"[대기열 추가] {custom_id}: queue_id={queue_result['queue_id']}")
                    else:
                        logger.warning(f"[재시도 실패] {custom_id}: {retry_result['error']}")
                        validation_stats["retry_failed"] = validation_stats.get("retry_failed", 0) + 1

                        # 재처리 대기열에 추가
                        chunk_info = {
                            "batch_id": meta.get("batchId"),
                            "chunk_key": custom_id,
                            "product_id": meta.get("productId"),
                            "product_code": meta.get("productCode"),
                            "image_index": meta.get("imageIndex"),
                            "chunk_index": meta.get("chunkIndex"),
                            "total_chunks": meta.get("totalChunks"),
                            "chunk_width": expected_width,
                            "chunk_height": expected_height,
                            "original_chunk_path": meta.get("originalChunkPath")
                        }
                        queue_result = enqueue_failed_chunk(
                            chunk_info=chunk_info,
                            config=config,
                            error_reason=retry_result.get('error', 'Unknown error'),
                            supabase_url=config.get("supabaseUrl"),
                            supabase_key=config.get("supabaseKey")
                        )
                        if queue_result["success"]:
                            validation_stats["queued_for_retry"] = validation_stats.get("queued_for_retry", 0) + 1
                            logger.info(f"[대기열 추가] {custom_id}: queue_id={queue_result['queue_id']}")
                else:
                    logger.error(f"[재처리 불가] {custom_id}: 데이터 손상으로 재처리 불가")
                    validation_stats["unretryable"] = validation_stats.get("unretryable", 0) + 1

                validation_stats["validation_failed"] = validation_stats.get("validation_failed", 0) + 1
            else:
                validation_stats["validation_passed"] = validation_stats.get("validation_passed", 0) + 1
                # 번역 검증 결과 로깅 (디버깅용)
                trans_val = validation.get("translation_validation")
                if trans_val and trans_val.get("has_text"):
                    logger.debug(f"[검증 통과] {custom_id}: 타겟 언어 비율 {trans_val.get('target_lang_ratio', 0):.1%}")
            # ===== 검증 끝 =====

            image_key = f"{meta['productId']}_{meta['imageIndex']}"

            if image_key not in image_chunks:
                image_chunks[image_key] = {
                    "productId": meta["productId"],
                    "productCode": meta.get("productCode"),
                    "imageIndex": meta["imageIndex"],
                    "totalChunks": meta.get("totalChunks", 1),
                    "originalWidth": meta.get("chunkWidth"),  # 원본 이미지 너비
                    "chunks": []
                }

            image_chunks[image_key]["chunks"].append({
                "index": meta["chunkIndex"],
                "base64": translated_base64,
                "height": meta.get("chunkHeight", 2000)
            })

        logger.info(f"Grouped into {len(image_chunks)} images")
        logger.info(f"Validation stats: {validation_stats}")

        # 3. 각 이미지 병합 및 업로드
        results = []
        # config에서 가져오거나 환경변수 사용 (fallback)
        supabase_url = config.get("supabaseUrl") or SUPABASE_URL
        supabase_key = config.get("supabaseKey") or SUPABASE_SERVICE_KEY
        storage_bucket = config.get("storageBucket", "translated-images")
        table_name = config.get("tableName", "")
        target_lang_code = config.get("targetLangCode", "en")

        # 출력 포맷 설정 (실시간 처리와 동일하게)
        output_format = config.get("outputFormat", "WEBP").upper()
        output_quality = int(config.get("outputQuality", 100))
        output_extension = config.get("outputExtension", ".webp")

        # 포맷 매핑 (JPG -> JPEG for PIL)
        pil_format = output_format
        if pil_format == "JPG":
            pil_format = "JPEG"

        # MIME 타입 매핑
        mime_types = {
            "WEBP": "image/webp",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png"
        }
        mime_type = mime_types.get(output_format, "image/webp")

        logger.info(f"Supabase URL: {supabase_url}")
        logger.info(f"Supabase Key present: {bool(supabase_key)}")
        logger.info(f"Storage bucket: {storage_bucket}")
        logger.info(f"Output format: {output_format} (PIL: {pil_format}), Quality: {output_quality}")

        for image_key, image_data in image_chunks.items():
            try:
                # 청크 정렬
                sorted_chunks = sorted(image_data["chunks"], key=lambda x: x["index"])

                logger.info(f"Processing image {image_key}: {len(sorted_chunks)}/{image_data['totalChunks']} chunks")

                if len(sorted_chunks) != image_data["totalChunks"]:
                    logger.warning(f"Image {image_key}: Missing chunks ({len(sorted_chunks)}/{image_data['totalChunks']})")

                # 원본 이미지 너비 가져오기
                original_width = image_data.get("originalWidth")

                # 청크가 1개면 병합 불필요, 하지만 포맷 변환 및 리사이즈 필요
                if len(sorted_chunks) == 1:
                    chunk_image = base64_to_image(sorted_chunks[0]["base64"])
                    # 원본 너비로 리사이즈 (Gemini가 크기를 변경했을 경우)
                    if original_width and chunk_image.size[0] != original_width:
                        original_height = int(chunk_image.size[1] * original_width / chunk_image.size[0])
                        chunk_image = chunk_image.resize((original_width, original_height), Image.Resampling.LANCZOS)
                        logger.info(f"Resized single chunk to original width: {original_width}px")
                    merged_base64, _ = image_to_base64(chunk_image, format=pil_format, quality=output_quality)
                else:
                    # 청크 병합 후 출력 포맷 적용 (원본 너비로 리사이즈)
                    merged_image = merge_images(sorted_chunks, overlap=0, blend_height=50, target_width=original_width)
                    merged_base64, _ = image_to_base64(merged_image, format=pil_format, quality=output_quality)

                # Supabase Storage 업로드
                if supabase_url and supabase_key:
                    image_num = str(image_data["imageIndex"] + 1).zfill(2)
                    file_name = f"{table_name}/{target_lang_code}/{table_name}_ID{image_data['productId']}_{image_num}_{target_lang_code}{output_extension}"

                    upload_url = f"{supabase_url}/storage/v1/object/{storage_bucket}/{file_name}"

                    upload_response = requests.post(
                        upload_url,
                        headers={
                            "Authorization": f"Bearer {supabase_key}",
                            "Content-Type": mime_type,
                            "x-upsert": "true"
                        },
                        data=base64.b64decode(merged_base64),
                        timeout=120
                    )

                    if upload_response.status_code in [200, 201]:
                        public_url = f"{supabase_url}/storage/v1/object/public/{storage_bucket}/{file_name}"
                        upload_error = None
                        logger.info(f"Uploaded: {file_name}")
                    else:
                        public_url = None
                        upload_error = f"HTTP {upload_response.status_code}: {upload_response.text[:200]}"
                        logger.error(f"Upload failed for {file_name}: {upload_error}")
                else:
                    public_url = None
                    upload_error = "No Supabase credentials provided"

                results.append({
                    "productId": image_data["productId"],
                    "productCode": image_data["productCode"],
                    "imageIndex": image_data["imageIndex"],
                    "chunksProcessed": len(sorted_chunks),
                    "totalChunks": image_data["totalChunks"],
                    "uploadedUrl": public_url,
                    "success": public_url is not None,
                    "uploadError": upload_error
                })

            except Exception as img_err:
                logger.error(f"Error processing image {image_key}: {img_err}")
                import traceback
                traceback.print_exc()
                results.append({
                    "productId": image_data["productId"],
                    "imageIndex": image_data["imageIndex"],
                    "error": str(img_err),
                    "success": False
                })

        # 4. 상품별 URL 정리
        product_urls = {}
        for result in results:
            if result.get("success") and result.get("uploadedUrl"):
                pid = str(result["productId"])
                if pid not in product_urls:
                    product_urls[pid] = []
                product_urls[pid].append({
                    "index": result["imageIndex"],
                    "url": result["uploadedUrl"]
                })

        # 인덱스 순 정렬
        for pid in product_urls:
            product_urls[pid] = sorted(product_urls[pid], key=lambda x: x["index"])

        success_count = sum(1 for r in results if r.get("success"))

        return jsonify({
            "success": True,
            "totalImages": len(image_chunks),
            "processedImages": len(results),
            "successCount": success_count,
            "failCount": len(results) - success_count,
            "validationStats": validation_stats,
            "results": results,
            "productUrls": product_urls
        })

    except Exception as e:
        logger.error(f"Error in process_batch_results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/translate-chunks", methods=["POST"])
def translate_chunks():
    """
    실시간 청크 번역 API - n8n 클라우드의 60초 타임아웃 우회용

    Request:
        chunks: 슬라이스된 청크 리스트 [{base64, height, y_start, y_end, index}, ...]
        gemini_api_key: Gemini API 키
        gemini_model: 사용할 모델 (기본: gemini-3-pro-image-preview)
        prompt: 번역 프롬프트

    Response:
        success: 성공 여부
        translated_chunks: 번역된 청크 리스트
        success_count: 성공한 청크 수
        fail_count: 실패한 청크 수
    """
    import time

    try:
        data = request.get_json()

        chunks = data.get("chunks", [])
        gemini_api_key = data.get("gemini_api_key")
        gemini_model = data.get("gemini_model", "gemini-3-pro-image-preview")
        prompt = data.get("prompt", "Translate the text in this image to English.")

        if not chunks:
            return jsonify({"error": "chunks required"}), 400
        if not gemini_api_key:
            return jsonify({"error": "gemini_api_key required"}), 400

        logger.info(f"[translate-chunks] Starting: {len(chunks)} chunks, model: {gemini_model}")

        # Exponential Backoff 설정
        MAX_RETRIES = 5
        BASE_DELAY = 1.0
        MAX_DELAY = 60.0

        translated_chunks = []
        progress_log = []

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            chunk_base64 = chunk.get("base64", "")
            chunk_height = chunk.get("height", 0)

            progress_log.append(f"[번역 {i+1}/{len(chunks)}] 청크 높이: {chunk_height}px")

            translated_base64 = chunk_base64  # 기본값: 원본
            success = False
            error_msg = ""
            retries = 0

            for attempt in range(MAX_RETRIES):
                try:
                    # Gemini API 요청
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"

                    request_body = {
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": chunk_base64}}
                            ]
                        }],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"]
                        }
                    }

                    response = requests.post(
                        api_url,
                        json=request_body,
                        headers={"Content-Type": "application/json"},
                        timeout=180
                    )

                    if response.status_code == 429 or response.status_code >= 500:
                        # Rate limit 또는 서버 에러 - 재시도
                        retries = attempt + 1
                        delay = min(BASE_DELAY * (2 ** attempt) + (time.time() % 1), MAX_DELAY)
                        progress_log.append(f"  -> {response.status_code} 발생, {delay:.1f}초 후 재시도")
                        time.sleep(delay)
                        continue

                    response.raise_for_status()
                    result = response.json()

                    # 응답에서 이미지 추출
                    candidates = result.get("candidates", [])
                    if candidates and candidates[0].get("content", {}).get("parts"):
                        for part in candidates[0]["content"]["parts"]:
                            inline_data = part.get("inline_data") or part.get("inlineData")
                            if inline_data and inline_data.get("data") and len(inline_data["data"]) > 1000:
                                translated_base64 = inline_data["data"]
                                success = True
                                break

                    if success:
                        break

                    # 이미지 없이 응답 완료 - 재시도 불필요
                    if candidates:
                        finish_reason = candidates[0].get("finishReason", "")
                        if finish_reason == "SAFETY":
                            error_msg = "안전 필터 차단"
                            break
                        error_msg = f"이미지 응답 없음 (finishReason: {finish_reason})"
                    else:
                        error_msg = "응답에 candidates 없음"
                    break

                except requests.exceptions.Timeout:
                    error_msg = "API 타임아웃"
                    retries = attempt + 1
                    if attempt < MAX_RETRIES - 1:
                        delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                        time.sleep(delay)

                except Exception as e:
                    error_msg = str(e)
                    retries = attempt + 1

                    # 429 또는 5xx 에러인 경우 재시도
                    if "429" in error_msg or "500" in error_msg or "503" in error_msg:
                        if attempt < MAX_RETRIES - 1:
                            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                            time.sleep(delay)
                            continue
                    break

            elapsed = time.time() - chunk_start

            if success:
                progress_log.append(f"  -> ✓ 성공 ({elapsed:.1f}초{f', 재시도: {retries}' if retries > 0 else ''})")
            else:
                progress_log.append(f"  -> ✗ 실패: {error_msg} ({elapsed:.1f}초)")

            translated_chunks.append({
                "index": chunk.get("index", i),
                "base64": translated_base64,
                "height": chunk_height,
                "y_start": chunk.get("y_start", 0),
                "y_end": chunk.get("y_end", 0),
                "success": success,
                "retries": retries,
                "error": None if success else error_msg
            })

        success_count = sum(1 for c in translated_chunks if c["success"])
        fail_count = len(translated_chunks) - success_count

        logger.info(f"[translate-chunks] 완료: {success_count}/{len(chunks)} 성공")

        return jsonify({
            "success": True,
            "translated_chunks": translated_chunks,
            "total_chunks": len(chunks),
            "success_count": success_count,
            "fail_count": fail_count,
            "progress_log": progress_log
        })

    except Exception as e:
        logger.error(f"Error in translate_chunks: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


@app.route("/prepare-batch", methods=["POST"])
def prepare_batch():
    """
    대량 배치 처리를 위한 준비 엔드포인트

    n8n Cloud의 메모리 제한을 우회하기 위해 Railway에서 직접 Supabase 조회
    메모리 효율을 위해 JSONL을 임시 파일에 스트리밍

    Request:
        config: 설정 {
            tableName, geminiApiKey, geminiModel, prompt, chunkHeight,
            supabaseUrl, supabaseKey, storageBucket, targetLangCode,
            outputFormat, outputExtension, outputQuality, limit
        }

    Response:
        success: 성공 여부
        batchName: Gemini 배치 작업명
        batchState: 배치 상태
        totalProducts: 총 상품 수
        totalImages: 총 이미지 수
        totalChunks: 총 청크 수
        chunkMetadata: 청크 메타데이터
        productMap: 상품 맵
    """
    import time
    import tempfile
    import gc

    temp_file = None

    try:
        data = request.get_json()
        config = data.get("config", {})

        # 필수 설정 확인
        gemini_api_key = config.get("geminiApiKey")
        if not gemini_api_key:
            return jsonify({"error": "geminiApiKey required in config"}), 400

        supabase_url = config.get("supabaseUrl")
        supabase_key = config.get("supabaseKey")
        table_name = config.get("tableName")

        if not supabase_url or not supabase_key:
            return jsonify({"error": "supabaseUrl and supabaseKey required in config"}), 400

        if not table_name:
            return jsonify({"error": "tableName required in config"}), 400

        gemini_model = config.get("geminiModel", "gemini-3-pro-image-preview")
        prompt = config.get("prompt", "Translate the text in this image.")
        chunk_height = config.get("chunkHeight", 3000)
        limit = config.get("limit", 1000)
        text_service_url = config.get("textServiceUrl", "https://text-render-service-production.up.railway.app")

        # 쿼리 타입 및 ID 필터링 옵션
        query_type = config.get("queryType", "limit")
        id_list = config.get("idList", [])
        id_start = config.get("idStart")
        id_end = config.get("idEnd")

        logger.info("=" * 60)
        logger.info(f"[prepare-batch] 시작")
        logger.info(f"[prepare-batch] 테이블: {table_name}")
        logger.info(f"[prepare-batch] 모델: {gemini_model}")
        logger.info(f"[prepare-batch] 청크 높이: {chunk_height}")
        logger.info(f"[prepare-batch] 쿼리 타입: {query_type}")
        if query_type == "ids":
            logger.info(f"[prepare-batch] ID 목록: {id_list}")
        elif query_type == "range":
            logger.info(f"[prepare-batch] ID 범위: {id_start} ~ {id_end}")
        else:
            logger.info(f"[prepare-batch] 조회 제한: {limit}")

        # 1. Supabase에서 직접 상품 조회
        logger.info(f"[prepare-batch] Supabase에서 상품 조회 중...")
        supabase_headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }

        # 쿼리 타입에 따른 URL 구성
        base_url = f"{supabase_url}/rest/v1/{table_name}"
        base_params = "select=id,product_code,description_images&description_images=not.is.null&order=id"

        if query_type == "ids" and id_list:
            id_filter = f"id=in.({','.join(map(str, id_list))})"
            url = f"{base_url}?{base_params}&{id_filter}"
            logger.info(f"[prepare-batch] ID 목록 필터: {id_filter}")
        elif query_type == "range" and id_start is not None and id_end is not None:
            id_filter = f"id=gte.{id_start}&id=lte.{id_end}"
            url = f"{base_url}?{base_params}&{id_filter}"
            logger.info(f"[prepare-batch] ID 범위 필터: {id_filter}")
        else:
            url = f"{base_url}?{base_params}&limit={limit}"
            logger.info(f"[prepare-batch] 제한 필터: limit={limit}")

        products_response = requests.get(
            url,
            headers=supabase_headers,
            timeout=60
        )

        if products_response.status_code >= 400:
            return jsonify({
                "error": f"Supabase 상품 조회 실패: HTTP {products_response.status_code}",
                "details": products_response.text[:500]
            }), 500

        products = products_response.json()
        logger.info(f"[prepare-batch] 번역 필요한 상품 수: {len(products)}")

        if not products:
            return jsonify({
                "success": True,
                "message": "번역이 필요한 상품이 없습니다",
                "totalProducts": 0,
                "totalImages": 0,
                "totalChunks": 0
            })

        # 1. 이미지 URL 수집 및 상품 맵 생성
        image_requests = []
        product_map = {}

        for product in products:
            product_id = product.get("id")
            product_code = product.get("product_code", "")
            desc_images = product.get("description_images", [])

            # description_images가 문자열이면 JSON 파싱
            if isinstance(desc_images, str):
                try:
                    desc_images = json.loads(desc_images)
                except:
                    desc_images = []

            if not desc_images:
                continue

            product_map[str(product_id)] = {
                "productCode": product_code,
                "imageCount": len(desc_images)
            }

            for i, image_url in enumerate(desc_images):
                if image_url and image_url.strip():
                    image_requests.append({
                        "productId": product_id,
                        "productCode": product_code,
                        "imageIndex": i,
                        "imageUrl": image_url,
                        "tableName": table_name
                    })

        logger.info(f"[prepare-batch] 총 이미지 수: {len(image_requests)}")

        if not image_requests:
            return jsonify({
                "success": False,
                "error": "처리할 이미지가 없습니다"
            }), 400

        # 배치 ID 생성 (원본 청크 저장용)
        batch_id = f"batch-{table_name}-{int(time.time())}"

        # 원본 청크 보존 설정 (기본: 활성화)
        preserve_original_chunks = config.get("preserveOriginalChunks", True)
        logger.info(f"[prepare-batch] 배치 ID: {batch_id}")
        logger.info(f"[prepare-batch] 원본 청크 보존: {preserve_original_chunks}")

        # 2. 임시 파일에 JSONL 스트리밍 (메모리 효율)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        chunk_metadata = []
        total_chunks = 0
        errors = []
        processed_images = 0
        saved_original_chunks = 0

        for img_idx, img_req in enumerate(image_requests):
            if img_idx % 50 == 0:
                logger.info(f"[prepare-batch] 진행: {img_idx + 1}/{len(image_requests)} 이미지")
                gc.collect()  # 주기적 가비지 컬렉션

            try:
                # 이미지 다운로드
                image_response = requests.get(
                    img_req["imageUrl"],
                    timeout=120
                )
                image_response.raise_for_status()

                image = Image.open(BytesIO(image_response.content)).convert("RGB")
                img_width, img_height = image.size

                # 이미지 슬라이스
                chunks = slice_image(image, chunk_height=chunk_height, use_smart_cut=True, overlap=0)

                # 각 청크에 대한 배치 요청 생성 및 파일에 직접 쓰기
                for chunk_idx, chunk in enumerate(chunks):
                    request_key = f"p{img_req['productId']}_i{img_req['imageIndex']}_c{chunk_idx}"

                    batch_request = {
                        "key": request_key,
                        "request": {
                            "contents": [{
                                "parts": [
                                    {"text": prompt},
                                    {"inline_data": {"mime_type": "image/jpeg", "data": chunk["base64"]}}
                                ]
                            }],
                            "generation_config": {
                                "response_modalities": ["TEXT", "IMAGE"]
                            }
                        }
                    }

                    # 파일에 직접 쓰기 (메모리 절약)
                    temp_file.write(json.dumps(batch_request) + '\n')

                    # 원본 청크 저장 (재처리용)
                    original_chunk_path = ""
                    if preserve_original_chunks:
                        save_result = save_original_chunk(
                            chunk_base64=chunk["base64"],
                            chunk_key=request_key,
                            batch_id=batch_id,
                            supabase_url=supabase_url,
                            supabase_key=supabase_key
                        )
                        if save_result["success"]:
                            original_chunk_path = save_result["path"]
                            saved_original_chunks += 1
                        else:
                            logger.warning(f"[prepare-batch] 원본 청크 저장 실패: {request_key} - {save_result['error']}")

                    chunk_metadata.append({
                        "key": request_key,
                        "productId": img_req["productId"],
                        "productCode": img_req["productCode"],
                        "imageIndex": img_req["imageIndex"],
                        "chunkIndex": chunk_idx,
                        "totalChunks": len(chunks),
                        "chunkHeight": chunk["height"],
                        "chunkWidth": img_width,  # 원본 이미지 너비 저장
                        "yStart": chunk["y_start"],
                        "yEnd": chunk["y_end"],
                        "originalChunkPath": original_chunk_path,  # 원본 청크 경로
                        "batchId": batch_id  # 배치 ID
                    })

                    total_chunks += 1

                # 메모리 해제
                del image
                del chunks
                processed_images += 1

            except Exception as img_err:
                error_msg = f"상품 {img_req['productId']} 이미지 {img_req['imageIndex']}: {str(img_err)}"
                logger.error(f"[prepare-batch] 에러: {error_msg}")
                errors.append(error_msg)

        temp_file.close()

        if total_chunks == 0:
            os.unlink(temp_file.name)
            return jsonify({
                "success": False,
                "error": "처리된 청크가 없습니다",
                "errors": errors
            }), 400

        logger.info("=" * 60)
        logger.info(f"[prepare-batch] 청크 생성 완료")
        logger.info(f"  - 처리된 이미지: {processed_images}")
        logger.info(f"  - 생성된 청크: {total_chunks}")
        logger.info(f"  - 저장된 원본 청크: {saved_original_chunks}")
        logger.info(f"  - 에러: {len(errors)}")

        # 3. Gemini Files API에 JSONL 업로드 (파일에서 직접 읽기)
        file_size = os.path.getsize(temp_file.name)
        logger.info(f"[prepare-batch] JSONL 파일 크기: {file_size} bytes")

        try:
            # 파일에서 직접 읽어서 업로드 (메모리 절약)
            with open(temp_file.name, 'rb') as f:
                jsonl_bytes = f.read()

            upload_response = requests.post(
                f'https://generativelanguage.googleapis.com/upload/v1beta/files?key={gemini_api_key}',
                headers={
                    'X-Goog-Upload-Command': 'upload, finalize',
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(len(jsonl_bytes))
                },
                data=jsonl_bytes,
                timeout=600
            )

            # 임시 파일 삭제
            os.unlink(temp_file.name)
            temp_file = None

            if upload_response.status_code >= 400:
                return jsonify({
                    "success": False,
                    "error": f"Files API 업로드 실패: HTTP {upload_response.status_code}",
                    "details": upload_response.text[:500]
                }), 500

            upload_result = upload_response.json()
            file_name = upload_result.get("file", {}).get("name")

            if not file_name:
                return jsonify({
                    "success": False,
                    "error": "Files API 응답에 파일명 없음",
                    "details": str(upload_result)[:500]
                }), 500

            logger.info(f"[prepare-batch] Files API 업로드 완료: {file_name}")

        except Exception as upload_err:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            return jsonify({
                "success": False,
                "error": f"Files API 업로드 에러: {str(upload_err)}"
            }), 500

        # 4. Gemini Batch API 제출
        # 배치 ID를 display_name으로 사용 (이미 생성됨)
        batch_display_name = batch_id

        try:
            batch_response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:batchGenerateContent',
                headers={
                    'Content-Type': 'application/json',
                    'x-goog-api-key': gemini_api_key
                },
                json={
                    "batch": {
                        "display_name": batch_display_name,
                        "input_config": {"file_name": file_name}
                    }
                },
                timeout=120
            )

            if batch_response.status_code >= 400:
                return jsonify({
                    "success": False,
                    "error": f"Batch API 제출 실패: HTTP {batch_response.status_code}",
                    "details": batch_response.text[:500]
                }), 500

            batch_result = batch_response.json()
            batch_name = batch_result.get("name", "")
            batch_state = batch_result.get("metadata", {}).get("state") or batch_result.get("state", "JOB_STATE_PENDING")

            logger.info(f"[prepare-batch] Batch API 제출 완료")
            logger.info(f"  - 배치명: {batch_name}")
            logger.info(f"  - 상태: {batch_state}")

        except Exception as batch_err:
            return jsonify({
                "success": False,
                "error": f"Batch API 제출 에러: {str(batch_err)}"
            }), 500

        # 5. 결과 반환
        logger.info("=" * 60)

        return jsonify({
            "success": True,
            "batchId": batch_id,  # 원본 청크 조회용 배치 ID
            "batchName": batch_name,
            "batchState": batch_state,
            "batchDisplayName": batch_display_name,
            "fileName": file_name,
            "totalProducts": len(product_map),
            "totalImages": len(image_requests),
            "totalChunks": total_chunks,
            "savedOriginalChunks": saved_original_chunks,  # 저장된 원본 청크 수
            "chunkMetadata": chunk_metadata,
            "productMap": product_map,
            "config": {
                "geminiApiKey": gemini_api_key,
                "geminiModel": gemini_model,
                "prompt": prompt,
                "chunkHeight": chunk_height,
                "supabaseUrl": config.get("supabaseUrl"),
                "supabaseKey": config.get("supabaseKey"),
                "storageBucket": config.get("storageBucket"),
                "textServiceUrl": text_service_url,
                "targetLangCode": config.get("targetLangCode", "en"),
                "outputFormat": config.get("outputFormat", "WEBP"),
                "outputExtension": config.get("outputExtension", ".webp"),
                "outputQuality": config.get("outputQuality", 100),
                "preserveOriginalChunks": preserve_original_chunks
            },
            "submittedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "errors": errors if errors else None
        })

    except Exception as e:
        logger.error(f"Error in prepare_batch: {e}")
        import traceback
        traceback.print_exc()
        # 임시 파일 정리
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting Text Render Service v9 on port {port}")
    logger.info(f"Features: slice, merge, batch-results, translate-chunks, prepare-batch")
    logger.info(f"Vertex AI available: {vertex_ai_available}")
    app.run(host="0.0.0.0", port=port, debug=True)
