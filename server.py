"""
텍스트 번역 및 렌더링 서비스 v10
- 긴 이미지 슬라이스/병합 지원
- Gemini 배치 처리 결과 처리
- 청크 번역 및 검증
- OCR 기반 번역 언어 검증
- v10: 이미지 단위 처리, 청크별 검증, 재처리 루프
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

# OCR Reader 단일 캐시 (메모리 절약: 항상 1개 언어 모델만 유지)
_current_ocr_reader = None
_current_ocr_lang = None

def get_ocr_reader(target_lang: str):
    """
    대상 언어에 맞는 PaddleOCR Reader를 반환 (단일 캐시)

    메모리 절약을 위해 한 번에 1개 언어 모델만 유지.
    다른 언어 요청 시 기존 모델 해제 후 새 모델 로딩.
    """
    global _current_ocr_reader, _current_ocr_lang
    import gc

    paddle_lang = LANGUAGE_CODE_MAP.get(target_lang, "en")

    if _current_ocr_lang == paddle_lang and _current_ocr_reader is not None:
        return _current_ocr_reader

    # 기존 모델 해제
    if _current_ocr_reader is not None:
        logger.info(f"PaddleOCR Reader 교체: {_current_ocr_lang} → {paddle_lang}")
        del _current_ocr_reader
        _current_ocr_reader = None
        _current_ocr_lang = None
        gc.collect()

    try:
        import logging as _logging
        _logging.getLogger('ppocr').setLevel(_logging.WARNING)

        well_supported_langs = {"ch", "chinese_cht", "en", "japan"}

        if paddle_lang in well_supported_langs:
            _current_ocr_reader = PaddleOCR(
                lang=paddle_lang,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=736,
                text_det_limit_type='max',
                text_det_box_thresh=0.5,
                text_det_thresh=0.3,
                text_det_unclip_ratio=1.6,
                text_rec_score_thresh=0.3,
            )
            logger.info(f"PaddleOCR Reader 생성 (mobile_rec): {paddle_lang}")
        else:
            _current_ocr_reader = PaddleOCR(
                lang=paddle_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=736,
                text_det_limit_type='max',
                text_det_box_thresh=0.5,
                text_det_thresh=0.3,
                text_det_unclip_ratio=1.6,
                text_rec_score_thresh=0.3,
            )
            logger.info(f"PaddleOCR Reader 생성 (auto model): {paddle_lang}")

        _current_ocr_lang = paddle_lang

    except Exception as e:
        logger.error(f"PaddleOCR Reader 생성 실패: {e}")
        if _current_ocr_reader is None:
            _current_ocr_reader = PaddleOCR(
                lang="en",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=736,
                text_det_limit_type='max',
                text_det_box_thresh=0.5,
                text_det_thresh=0.3,
                text_det_unclip_ratio=1.6,
                text_rec_score_thresh=0.3,
            )
            _current_ocr_lang = "en"

    return _current_ocr_reader


def validate_translation_language(
    image_base64: str,
    target_lang: str,
    source_lang: str = None,
    threshold: float = 0.2,
    image_np=None
) -> dict:
    """
    OCR을 사용하여 번역된 이미지가 올바른 언어인지 검증 (이중 검사)

    검증 방식:
    1. source_lang이 제공된 경우: 원본 언어 리더로 "남아있으면 안 되는 텍스트" 검사
       → 원본 언어 텍스트가 threshold 이상 감지되면 번역 실패
    2. source_lang이 없는 경우: 타겟 언어 리더로 "있어야 하는 텍스트" 검사
       → 타겟 언어 비율이 (1-threshold) 미만이면 번역 실패

    Args:
        image_base64: 검증할 이미지의 base64 문자열
        target_lang: 번역 대상 언어 코드 (예: 'en', 'ja', 'zh')
        source_lang: 원본 언어 코드 (예: 'ko', 'zh') - 이중 검사용
        threshold: 허용 비율 (기본 20%)
            - source_lang 있음: 원본 언어 허용 비율 (20% 이하만 통과)
            - source_lang 없음: 비타겟 언어 허용 비율
        image_np: 이미 디코딩된 numpy array (있으면 base64 디코딩 스킵)

    Returns:
        dict: {
            "valid": bool,
            "reason": str,
            "has_text": bool,
            "total_chars": int,
            "source_lang_ratio": float,  # 원본 언어 비율 (source_lang 사용시)
            "target_lang_ratio": float,  # 타겟 언어 비율
            "detected_text": list
        }
    """
    try:
        # 1. 이미지 디코딩 (image_np가 없을 때만)
        if image_np is None:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            del image_bytes

            # RGB 변환 (필요한 경우)
            if len(image_np.shape) == 2:  # 그레이스케일
                pass  # PaddleOCR은 그레이스케일도 처리 가능
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # ================================================================
        # 이중 검사: source_lang이 제공된 경우 원본 언어 리더로 검사
        # ================================================================
        if source_lang:
            return _validate_by_source_lang(image_np, source_lang, target_lang, threshold)

        # ================================================================
        # 기존 방식: 타겟 언어 리더로 검사
        # ================================================================
        return _validate_by_target_lang(image_np, target_lang, threshold)

    except Exception as e:
        logger.error(f"번역 언어 검증 오류: {e}")
        # 오류 발생시 통과 처리 (false positive 방지)
        return {
            "valid": True,
            "reason": f"검증 오류로 통과 처리: {str(e)}",
            "has_text": False,
            "total_chars": 0,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": []
        }


def _validate_by_source_lang(image_np, source_lang: str, target_lang: str, threshold: float) -> dict:
    """
    원본 언어 리더로 "남아있으면 안 되는 텍스트" 검사

    원본 언어 텍스트가 많이 감지되면 = 번역 실패
    """
    # 원본 언어 OCR 실행
    reader = get_ocr_reader(source_lang)
    results = reader.predict(image_np)

    if results is None:
        return {
            "valid": True,
            "reason": "텍스트 없음 - 검증 통과",
            "has_text": False,
            "total_chars": 0,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": []
        }

    # OCR 결과 파싱
    detected_texts = []
    total_chars = 0

    try:
        for res in results:
            inner_res = res.json.get('res', {}) if hasattr(res.json, 'get') else {}
            rec_texts = inner_res.get('rec_texts', []) if hasattr(inner_res, 'get') else []
            rec_scores = inner_res.get('rec_scores', []) if hasattr(inner_res, 'get') else []

            for text, confidence in zip(rec_texts, rec_scores):
                if confidence is None or confidence <= 0.3:
                    continue
                detected_texts.append({
                    "text": str(text),
                    "confidence": float(confidence),
                    "char_count": len(str(text).replace(" ", ""))
                })
                total_chars += len(str(text).replace(" ", ""))
    except Exception as parse_error:
        logger.error(f"OCR 결과 파싱 오류: {parse_error}")
        return {
            "valid": True,
            "reason": f"OCR 파싱 오류로 통과 처리: {str(parse_error)}",
            "has_text": False,
            "total_chars": 0,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": []
        }

    # 문자 분류: 숫자/기호/구두점은 언어 중립이므로 비율 계산에서 제외
    all_text = "".join([t["text"] for t in detected_texts])
    letter_chars = sum(1 for c in all_text if c.isalpha())

    # 텍스트가 너무 적으면 통과 (번역 성공으로 간주)
    if letter_chars < 5:
        return {
            "valid": True,
            "reason": f"원본 언어({source_lang}) 텍스트 거의 없음 - 번역 성공",
            "has_text": total_chars > 0,
            "total_chars": total_chars,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": detected_texts
        }

    # 원본 언어 문자 카운트 (letter_chars 기준 비율)
    source_lang_chars = count_target_language_chars(all_text, source_lang)
    source_ratio = source_lang_chars / letter_chars if letter_chars > 0 else 0.0

    # 검증: 원본 언어 비율이 threshold 초과하면 번역 실패
    if source_ratio > threshold:
        return {
            "valid": False,
            "reason": f"번역 미완료: 원본 언어({source_lang}) {source_ratio:.1%} 남음 (허용 {threshold:.0%})",
            "has_text": True,
            "total_chars": total_chars,
            "source_lang_ratio": source_ratio,
            "target_lang_ratio": 1.0 - source_ratio,
            "detected_text": detected_texts
        }

    return {
        "valid": True,
        "reason": f"번역 검증 통과: 원본 언어({source_lang}) {source_ratio:.1%}만 남음",
        "has_text": True,
        "total_chars": total_chars,
        "source_lang_ratio": source_ratio,
        "target_lang_ratio": 1.0 - source_ratio,
        "detected_text": detected_texts
    }


def _validate_by_target_lang(image_np, target_lang: str, threshold: float) -> dict:
    """
    타겟 언어 리더로 "있어야 하는 텍스트" 검사 (기존 방식)
    """
    reader = get_ocr_reader(target_lang)
    results = reader.predict(image_np)

    if results is None:
        return {
            "valid": True,
            "reason": "텍스트 없음 - 검증 통과",
            "has_text": False,
            "total_chars": 0,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": []
        }

    # OCR 결과 파싱
    detected_texts = []
    total_chars = 0

    try:
        for res in results:
            inner_res = res.json.get('res', {}) if hasattr(res.json, 'get') else {}
            rec_texts = inner_res.get('rec_texts', []) if hasattr(inner_res, 'get') else []
            rec_scores = inner_res.get('rec_scores', []) if hasattr(inner_res, 'get') else []

            for text, confidence in zip(rec_texts, rec_scores):
                if confidence is None or confidence <= 0.3:
                    continue
                detected_texts.append({
                    "text": str(text),
                    "confidence": float(confidence),
                    "char_count": len(str(text).replace(" ", ""))
                })
                total_chars += len(str(text).replace(" ", ""))
    except Exception as parse_error:
        logger.error(f"OCR 결과 파싱 오류: {parse_error}")
        return {
            "valid": True,
            "reason": f"OCR 파싱 오류로 통과 처리: {str(parse_error)}",
            "has_text": False,
            "total_chars": 0,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": []
        }

    # 문자 분류: 숫자/기호/구두점은 언어 중립이므로 비율 계산에서 제외
    all_text = "".join([t["text"] for t in detected_texts])
    letter_chars = sum(1 for c in all_text if c.isalpha())

    # 텍스트가 너무 적으면 통과
    if letter_chars < 5:
        return {
            "valid": True,
            "reason": "텍스트 적음 - 검증 통과",
            "has_text": total_chars > 0,
            "total_chars": total_chars,
            "source_lang_ratio": 0.0,
            "target_lang_ratio": 1.0,
            "detected_text": detected_texts
        }

    # 타겟 언어 문자 카운트 (letter_chars 기준 비율)
    target_lang_chars = count_target_language_chars(all_text, target_lang)
    target_ratio = target_lang_chars / letter_chars if letter_chars > 0 else 1.0
    non_target_ratio = 1.0 - target_ratio

    # 검증: 비타겟 언어 비율이 threshold 초과하면 번역 실패
    if non_target_ratio > threshold:
        return {
            "valid": False,
            "reason": f"번역 미완료: 타겟 언어({target_lang}) 비율 {target_ratio:.1%}",
            "has_text": True,
            "total_chars": total_chars,
            "source_lang_ratio": non_target_ratio,
            "target_lang_ratio": target_ratio,
            "detected_text": detected_texts
        }

    return {
        "valid": True,
        "reason": f"번역 검증 통과: 타겟 언어({target_lang}) 비율 {target_ratio:.1%}",
        "has_text": True,
        "total_chars": total_chars,
        "source_lang_ratio": 1.0 - target_ratio,
        "target_lang_ratio": target_ratio,
        "detected_text": detected_texts
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


def resize_chunk_to_original(chunk_base64, expected_width, expected_height):
    """
    Gemini 출력 이미지를 원본 크기로 리사이즈

    Gemini API는 입력과 다른 해상도로 이미지를 반환할 수 있음.
    검증/병합 전에 원본 크기로 맞춰야 정확한 처리 가능.

    Returns:
        tuple: (resized_base64, was_resized, actual_width, actual_height)
    """
    try:
        image_bytes = base64.b64decode(chunk_base64)
        image = Image.open(BytesIO(image_bytes))
        actual_width, actual_height = image.size

        target_w = expected_width or actual_width
        target_h = expected_height or actual_height

        if actual_width == target_w and actual_height == target_h:
            del image_bytes
            image.close()
            return chunk_base64, False, actual_width, actual_height

        # LANCZOS 리사이즈 (merge_images와 동일 알고리즘)
        image = image.convert("RGB")
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=95)
        resized_base64 = base64.b64encode(buf.getvalue()).decode()

        del image_bytes
        buf.close()
        image.close()

        return resized_base64, True, actual_width, actual_height
    except Exception as e:
        logger.warning(f"[ResizeChunk] Failed: {e}")
        return chunk_base64, False, 0, 0


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
        # 1. Base64 데이터 존재 여부
        if not chunk_base64:
            return {"valid": False, "reason": "이미지 데이터 없음", "actual_width": 0, "actual_height": 0}

        # 2. 이미지 디코딩 (실제 디코딩 성공 여부로 유효성 판단)
        try:
            image_bytes = base64.b64decode(chunk_base64)
            image = Image.open(BytesIO(image_bytes))
            image.verify()  # 이미지 무결성 검증
            # verify() 후 다시 열어야 함
            image = Image.open(BytesIO(image_bytes))
            actual_width, actual_height = image.size
            # OCR 재사용을 위해 numpy array 생성
            image_np = np.array(image)
            del image_bytes
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
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

        return {"valid": True, "reason": "검증 통과", "actual_width": actual_width, "actual_height": actual_height, "image_np": image_np}

    except Exception as e:
        return {"valid": False, "reason": f"검증 중 오류: {str(e)}", "actual_width": 0, "actual_height": 0}


def validate_translated_chunk(
    chunk_base64: str,
    target_lang: str = None,
    source_lang: str = None,
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
        source_lang: 원본 언어 코드 (예: 'ko', 'zh') - 이중 검사용
        expected_width: 예상 너비
        expected_height: 예상 높이
        size_tolerance: 크기 허용 오차 (기본 30%)
        lang_threshold: 원본 언어 허용 비율 (기본 20%)
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
    # image_np 추출 후 size_validation에서 제거 (JSON 직렬화 불가)
    chunk_image_np = size_result.pop("image_np", None)
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
                source_lang=source_lang,
                threshold=lang_threshold,
                image_np=chunk_image_np
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
        finally:
            # OCR 완료 후 numpy array 즉시 해제
            del chunk_image_np
    else:
        del chunk_image_np

    result["reason"] = "모든 검증 통과"
    return result


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
        "service": "text-render-service-v10",
        "vertex_ai_available": vertex_ai_available,
        "project_id": PROJECT_ID,
        "features": ["slice", "merge", "batch-results", "translate-chunks", "prepare-batch", "ocr-validation", "original-chunk-preservation", "retry-queue"]
    })


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
        gemini_chunks = 0  # Gemini에 보낼 청크 수
        skipped_no_text = 0  # 텍스트 없어서 스킵한 청크 수
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

                    # OOM 방지: OCR 텍스트 감지 비활성화
                    # 모든 청크를 Gemini에 전송하고, validation 단계에서만 OCR 사용
                    chunk_has_text = True  # OCR 비활성화 - 모든 청크 처리

                    # 모든 청크 → Gemini 배치 요청 생성
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
                    gemini_chunks += 1

                    # 원본 청크 저장 (재처리용 - 텍스트 유무와 관계없이)
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
                        "batchId": batch_id,  # 배치 ID
                        "hasText": chunk_has_text  # 텍스트 유무 플래그
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
        logger.info(f"  - 총 청크: {total_chunks}")
        logger.info(f"  - Gemini 전송: {gemini_chunks} (번역 필요)")
        logger.info(f"  - 텍스트 없음: {skipped_no_text} (원본 사용)")
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
            "geminiChunks": gemini_chunks,  # Gemini에 전송한 청크 수
            "skippedNoText": skipped_no_text,  # 텍스트 없어서 스킵한 청크 수
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
                "sourceLangCode": config.get("sourceLangCode", "ko"),
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


# ===== v10: 새로운 배치 처리 시스템 =====
from batch_processor import (
    get_invalid_chunks_for_retry,
    get_batch_progress,
    check_and_complete_images,
    update_chunk_status,
    update_image_status,
    supabase_request,
    MAX_RETRY_COUNT
)


@app.route("/create-retry-batch", methods=["POST"])
def create_retry_batch_endpoint():
    """
    실패한 청크들로 재처리 배치 생성

    Request:
        batch_job_id: 원본 배치 ID
        config: 설정 (geminiApiKey, geminiModel, prompt 등)

    Returns:
        success: 성공 여부
        retry_batch_name: 새 Gemini 배치 이름
        chunks_count: 재처리 청크 수
    """
    try:
        data = request.get_json() or {}

        batch_job_id = data.get("batch_job_id")
        config = data.get("config", {})

        if not batch_job_id:
            return jsonify({"error": "batch_job_id required"}), 400

        supabase_url = config.get("supabaseUrl") or SUPABASE_URL
        supabase_key = config.get("supabaseKey") or SUPABASE_SERVICE_KEY
        gemini_api_key = config.get("geminiApiKey")
        prompt = config.get("prompt", "Translate the text in this image.")
        gemini_model = config.get("geminiModel", "gemini-3-pro-image-preview")

        if not gemini_api_key:
            return jsonify({"error": "geminiApiKey required"}), 400

        # 재처리 대상 청크 조회
        success, invalid_chunks = get_invalid_chunks_for_retry(
            batch_job_id, supabase_url, supabase_key
        )

        if not success:
            return jsonify({"error": str(invalid_chunks)}), 500

        if not invalid_chunks:
            return jsonify({
                "success": True,
                "message": "No chunks to retry",
                "chunks_count": 0
            })

        logger.info(f"[RetryBatch] Found {len(invalid_chunks)} chunks to retry for job {batch_job_id}")

        # 3회 초과 청크는 원본으로 대체
        chunks_to_retry = []
        replaced_chunks = []

        for chunk in invalid_chunks:
            if chunk.get("retry_count", 0) >= MAX_RETRY_COUNT:
                # 3회 실패 → 원본으로 대체
                update_chunk_status(
                    chunk["id"], "replaced",
                    supabase_url, supabase_key
                )
                replaced_chunks.append(chunk["id"])
            else:
                chunks_to_retry.append(chunk)

        if not chunks_to_retry:
            # 모두 원본으로 대체됨 → 이미지 완료 체크
            complete_result = check_and_complete_images(
                batch_job_id, config,
                base64_to_image, image_to_base64, merge_images
            )
            return jsonify({
                "success": True,
                "message": "All chunks replaced with originals",
                "replaced_count": len(replaced_chunks),
                "completed_images": complete_result.get("completed", 0)
            })

        # batch_jobs에서 chunk_metadata 조회 (batchId는 chunk_metadata 안에 있음)
        from batch_processor import supabase_request as bp_supabase_request
        success, jobs = bp_supabase_request(
            "GET",
            f"batch_jobs?id=eq.{batch_job_id}&select=chunk_metadata",
            supabase_url, supabase_key
        )

        storage_batch_id = ""
        chunk_meta_map = {}
        if success and jobs:
            chunk_metadata = jobs[0].get("chunk_metadata", [])
            # chunk key -> metadata 매핑 & batchId 추출
            for meta in chunk_metadata:
                chunk_meta_map[f"{meta.get('productId')}_{meta.get('imageIndex')}_{meta.get('chunkIndex')}"] = meta
                # batchId는 모든 청크에 동일하므로 첫 번째에서 추출
                if not storage_batch_id and meta.get("batchId"):
                    storage_batch_id = meta.get("batchId")

        logger.info(f"[RetryBatch] Storage batch_id from chunk_metadata: {storage_batch_id}")

        # Gemini 배치 요청 생성
        import tempfile
        import time

        batch_requests = []
        chunk_ids = []
        skipped_no_data = 0

        for chunk in chunks_to_retry:
            original_b64 = chunk.get("original_base64", "")

            # 청크 메타 추출 (키 생성 + Storage 조회용)
            img_info = chunk.get("image_processing", {})
            product_id = img_info.get("product_id")
            image_index = img_info.get("image_index")
            chunk_index = chunk.get("chunk_index")

            # original_base64가 없으면 Storage에서 가져오기
            if not original_b64 and storage_batch_id:
                if product_id is not None and image_index is not None and chunk_index is not None:
                    chunk_key = f"p{product_id}_i{image_index}_c{chunk_index}"
                    logger.info(f"[RetryBatch] Fetching from storage: {chunk_key}")

                    # Storage에서 원본 가져오기
                    original_result = get_original_chunk(
                        chunk_key=chunk_key,
                        batch_id=storage_batch_id,
                        supabase_url=supabase_url,
                        supabase_key=supabase_key
                    )

                    if original_result.get("success"):
                        original_b64 = original_result.get("base64", "")
                        logger.info(f"[RetryBatch] Fetched {chunk_key} from storage: {len(original_b64)} chars")

            if not original_b64:
                logger.warning(f"[RetryBatch] Chunk {chunk['id']} has no original_base64 and not in storage")
                skipped_no_data += 1
                continue

            # 원본 키 형식 사용 (/get-batch-images의 chunk_metadata 매칭 호환)
            if product_id is not None and image_index is not None and chunk_index is not None:
                custom_id = f"p{product_id}_i{image_index}_c{chunk_index}"
            else:
                custom_id = f"retry_{batch_job_id}_{chunk['id']}_{int(time.time())}"

            batch_requests.append({
                "key": custom_id,
                "request": {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": original_b64}}
                        ]
                    }],
                    "generation_config": {
                        "response_modalities": ["TEXT", "IMAGE"]
                    }
                }
            })
            chunk_ids.append(chunk["id"])

        if not batch_requests:
            return jsonify({
                "success": False,
                "error": "No valid chunks to retry",
                "skipped_no_data": skipped_no_data,
                "total_chunks_found": len(chunks_to_retry)
            })

        # JSONL 파일 생성
        jsonl_content = "\n".join(json.dumps(req) for req in batch_requests)

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        temp_file.write(jsonl_content)
        temp_file.close()

        try:
            # Gemini 파일 업로드
            with open(temp_file.name, 'rb') as f:
                upload_response = requests.post(
                    "https://generativelanguage.googleapis.com/upload/v1beta/files",
                    headers={"x-goog-api-key": gemini_api_key},
                    files={"file": ("retry_batch.jsonl", f, "application/jsonl")},
                    timeout=120
                )

            if upload_response.status_code != 200:
                return jsonify({
                    "error": f"File upload failed: {upload_response.status_code}",
                    "details": upload_response.text[:500]
                }), 400

            file_info = upload_response.json()
            file_uri = file_info.get("file", {}).get("uri")

            if not file_uri:
                return jsonify({"error": "No file URI in response"}), 400

            # 배치 작업 생성
            file_name = file_info.get("file", {}).get("name")
            retry_batch_display_name = f"retry-{batch_job_id}-{int(time.time())}"

            batch_response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:batchGenerateContent",
                headers={
                    "x-goog-api-key": gemini_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "batch": {
                        "display_name": retry_batch_display_name,
                        "input_config": {"file_name": file_name}
                    }
                },
                timeout=60
            )

            if batch_response.status_code != 200:
                return jsonify({
                    "error": f"Batch creation failed: {batch_response.status_code}",
                    "details": batch_response.text[:500]
                }), 400

            batch_info = batch_response.json()
            batch_name = batch_info.get("name", "")

            # 청크 상태를 retrying으로 업데이트
            for chunk_id in chunk_ids:
                update_chunk_status(
                    chunk_id, "retrying",
                    supabase_url, supabase_key,
                    increment_retry=True
                )

            # batch_jobs에 재처리 배치 정보 기록
            supabase_request(
                "PATCH",
                f"batch_jobs?id=eq.{batch_job_id}",
                supabase_url, supabase_key,
                {
                    "retry_batch_name": batch_name,
                    "retry_count": len(chunk_ids)
                }
            )

            return jsonify({
                "success": True,
                "retry_batch_name": batch_name,
                "chunks_count": len(chunk_ids),
                "replaced_count": len(replaced_chunks),
                "chunk_ids": chunk_ids
            })

        finally:
            os.unlink(temp_file.name)

    except Exception as e:
        logger.error(f"[RetryBatch] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/complete-batch/<int:batch_id>", methods=["POST"])
def complete_batch(batch_id):
    """
    배치 완료 처리 및 상품 테이블 업데이트

    모든 이미지가 completed 상태일 때 호출
    각 상품의 translated_image 필드를 업데이트합니다.
    """
    try:
        data = request.get_json() or {}
        config = data.get("config", {})

        supabase_url = config.get("supabaseUrl") or SUPABASE_URL
        supabase_key = config.get("supabaseKey") or SUPABASE_SERVICE_KEY
        table_name = config.get("tableName", "")

        # tableName이 config에 없으면 batch_jobs.table_name에서 조회
        if not table_name:
            success, batch_records = supabase_request(
                "GET",
                f"batch_jobs?id=eq.{batch_id}&select=table_name",
                supabase_url, supabase_key
            )
            if success and batch_records:
                table_name = batch_records[0].get("table_name", "")
                if table_name:
                    logger.info(f"[CompleteBatch] tableName from batch_jobs.table_name: {table_name}")

        if not table_name:
            return jsonify({"error": "tableName required in config"}), 400

        # batch_jobs에서 전체 config 조회 (check_and_complete_images에 필요)
        merge_config = dict(config)
        if not merge_config.get("supabaseUrl"):
            success_bj, bj_records = supabase_request(
                "GET",
                f"batch_jobs?id=eq.{batch_id}&select=config",
                supabase_url, supabase_key
            )
            if success_bj and bj_records and bj_records[0].get("config"):
                merge_config = bj_records[0]["config"]
        # fallback 보장
        merge_config.setdefault("supabaseUrl", supabase_url)
        merge_config.setdefault("supabaseKey", supabase_key)
        merge_config.setdefault("tableName", table_name)

        # partial 이미지 병합 시도 (청크가 모두 valid/replaced면 completed로 전환)
        try:
            merge_result = check_and_complete_images(
                batch_id, merge_config,
                base64_to_image, image_to_base64, merge_images
            )
            if merge_result.get("completed", 0) > 0:
                logger.info(f"[CompleteBatch] Merged {merge_result['completed']} images")
        except Exception as e:
            logger.warning(f"[CompleteBatch] check_and_complete_images error: {e}")
            import traceback
            traceback.print_exc()

        # 진행 상황 확인
        progress = get_batch_progress(batch_id, supabase_url, supabase_key)

        if not progress.get("is_complete"):
            return jsonify({
                "success": False,
                "message": "Batch not complete yet",
                "progress": progress
            })

        # 상품별 이미지 URL 수집
        product_urls = {}

        for img in progress.get("images", []):
            if img["status"] != "completed":
                continue

            product_id = img["product_id"]

            # merged_url 조회
            success, img_records = supabase_request(
                "GET",
                f"image_processing?id=eq.{img['id']}&select=merged_url",
                supabase_url, supabase_key
            )

            if success and img_records and img_records[0].get("merged_url"):
                if product_id not in product_urls:
                    product_urls[product_id] = []

                product_urls[product_id].append({
                    "index": img["image_index"],
                    "url": img_records[0]["merged_url"]
                })

        # 상품별 업데이트
        updated_products = []

        for product_id, urls in product_urls.items():
            # 인덱스 순 정렬
            sorted_urls = [u["url"] for u in sorted(urls, key=lambda x: x["index"])]

            # 상품 테이블 업데이트
            success, result = supabase_request(
                "PATCH",
                f"{table_name}?id=eq.{product_id}",
                supabase_url, supabase_key,
                {"translated_image": sorted_urls}
            )

            if success:
                updated_products.append(product_id)

        # batch_jobs 완료 처리
        from datetime import datetime
        supabase_request(
            "PATCH",
            f"batch_jobs?id=eq.{batch_id}",
            supabase_url, supabase_key,
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat()
            }
        )

        return jsonify({
            "success": True,
            "updated_products": updated_products,
            "total_products": len(updated_products)
        })

    except Exception as e:
        logger.error(f"[CompleteBatch] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# datetime import 추가 (파일 상단에 없으면)
from datetime import datetime


# ===== n8n 루프 기반 처리 엔드포인트 =====

@app.route("/get-batch-images", methods=["POST"])
def get_batch_images():
    """
    완료된 Gemini 배치에서 이미지별 청크 목록 반환 (n8n 루프용)

    n8n에서 이미지 단위로 순환하면서 처리할 수 있도록
    이미지별로 그룹화된 청크 데이터를 반환

    Request:
        gemini_file_name: Gemini 결과 파일명
        gemini_api_key: Gemini API 키
        chunk_metadata: 청크 메타데이터 리스트
        batch_job_id: 배치 작업 ID (선택, DB 업데이트용)

    Response:
        images: [{
            image_key: "productId_imageIndex",
            product_id: int,
            image_index: int,
            total_chunks: int,
            chunks: [{index, base64, width, height, key}, ...]
        }, ...]
    """
    try:
        data = request.get_json()

        gemini_file_name = data.get("gemini_file_name")
        gemini_api_key = data.get("gemini_api_key")
        chunk_metadata = data.get("chunk_metadata", [])
        batch_job_id = data.get("batch_job_id")
        limit = data.get("limit")  # 반환할 이미지 수 제한 (메모리 절약)
        offset = data.get("offset", 0)  # 시작 위치

        if not gemini_file_name or not gemini_api_key:
            return jsonify({"error": "gemini_file_name and gemini_api_key required"}), 400

        logger.info(f"[GetBatchImages] Processing: {gemini_file_name}")
        logger.info(f"[GetBatchImages] Chunk metadata count: {len(chunk_metadata)}")

        # 1. Gemini 결과 파일 다운로드
        download_url = f"https://generativelanguage.googleapis.com/download/v1beta/{gemini_file_name}:download?alt=media"

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
        logger.info(f"[GetBatchImages] Downloaded {len(jsonl_content)} bytes")

        # 2. JSONL 파싱
        lines = jsonl_content.strip().split('\n')

        # custom_id로 청크 메타데이터 인덱싱
        meta_by_key = {m["key"]: m for m in chunk_metadata}

        # 이미지별로 청크 그룹핑
        image_chunks = {}

        for line_num, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                response_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = response_data.get("key") or response_data.get("custom_id")
            if not custom_id:
                continue

            meta = meta_by_key.get(custom_id)
            if not meta:
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
                logger.warning(f"[GetBatchImages] No image data for {custom_id}")
                continue

            image_key = f"{meta['productId']}_{meta['imageIndex']}"

            if image_key not in image_chunks:
                image_chunks[image_key] = {
                    "image_key": image_key,
                    "product_id": meta["productId"],
                    "image_index": meta["imageIndex"],
                    "total_chunks": meta.get("totalChunks", 1),
                    "original_width": meta.get("chunkWidth"),
                    "chunks": []
                }

            image_chunks[image_key]["chunks"].append({
                "index": meta["chunkIndex"],
                "base64": translated_base64,
                "width": meta.get("chunkWidth"),
                "height": meta.get("chunkHeight"),
                "key": custom_id,
                "has_text": meta.get("hasText", True)
            })

        # 이미지 리스트로 변환 (청크 정렬)
        all_images = []
        for image_key, image_data in image_chunks.items():
            image_data["chunks"] = sorted(image_data["chunks"], key=lambda x: x["index"])
            all_images.append(image_data)

        # 이미지 키 기준 정렬 (일관된 순서 보장)
        all_images = sorted(all_images, key=lambda x: (x["product_id"], x["image_index"]))

        total_images = len(all_images)
        logger.info(f"[GetBatchImages] Grouped into {total_images} images")

        # limit/offset 적용 (메모리 절약)
        if limit is not None:
            images = all_images[offset:offset + limit]
            has_more = (offset + limit) < total_images
            logger.info(f"[GetBatchImages] Returning {len(images)} images (offset={offset}, limit={limit}, has_more={has_more})")
        else:
            images = all_images
            has_more = False

        return jsonify({
            "success": True,
            "images": images,
            "total_images": total_images,
            "returned_count": len(images),
            "offset": offset,
            "has_more": has_more,
            "batch_job_id": batch_job_id
        })

    except Exception as e:
        logger.error(f"[GetBatchImages] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/validate-image-chunks", methods=["POST"])
def validate_image_chunks():
    """
    단일 이미지의 청크들을 검증 (n8n 루프용)

    이미지 단위로 청크를 검증하고 불량 청크 목록 반환

    Request:
        image_key: 이미지 식별자 ("productId_imageIndex")
        chunks: 청크 리스트 [{index, base64, width, height, key}, ...]
        target_lang: 번역 대상 언어 (예: "en")
        source_lang: 원본 언어 (예: "ko") - 이중 검사용, 권장
        skip_ocr: OCR 검증 건너뛰기 (기본 false)

    Response:
        image_key: 이미지 식별자
        all_valid: 모든 청크가 정상인지
        valid_chunks: 정상 청크 리스트
        defective_chunks: 불량 청크 리스트 [{index, key, reason, defect_type}, ...]
    """
    try:
        data = request.get_json()

        image_key = data.get("image_key")
        chunks = data.get("chunks", [])
        target_lang = data.get("target_lang", "en")
        source_lang = data.get("source_lang")  # 원본 언어 (이중 검사용)
        skip_ocr = data.get("skip_ocr", False)

        if not image_key or not chunks:
            return jsonify({"error": "image_key and chunks required"}), 400

        logger.info(f"[ValidateImageChunks] Validating {len(chunks)} chunks for {image_key}")

        import gc
        valid_chunks = []
        defective_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_index = chunk.get("index")
            chunk_base64 = chunk.get("base64")
            chunk_key = chunk.get("key")
            expected_width = chunk.get("width")
            expected_height = chunk.get("height")
            has_text = chunk.get("has_text", True)

            if not chunk_base64:
                defective_chunks.append({
                    "index": chunk_index,
                    "key": chunk_key,
                    "reason": "이미지 데이터 없음",
                    "defect_type": "missing_data",
                    "can_retry": False
                })
                continue

            # Gemini 출력 → 원본 크기로 리사이즈 (Gemini는 입력과 다른 해상도로 반환)
            if expected_width or expected_height:
                chunk_base64, was_resized, orig_w, orig_h = resize_chunk_to_original(
                    chunk_base64, expected_width, expected_height
                )
                if was_resized:
                    logger.info(f"[ValidateImageChunks] chunk {chunk_index}: resized {orig_w}x{orig_h} → {expected_width}x{expected_height}")

            # 텍스트 없는 청크는 검증 건너뛰기 (정상 처리)
            if not has_text:
                valid_chunks.append({
                    "index": chunk_index,
                    "key": chunk_key,
                    "base64": chunk_base64,
                    "reason": "텍스트 없음 - 검증 생략"
                })
                continue

            # 통합 검증 (크기 + OCR)
            validation = validate_translated_chunk(
                chunk_base64,
                target_lang=target_lang if not skip_ocr else None,
                source_lang=source_lang if not skip_ocr else None,
                expected_width=expected_width,
                expected_height=expected_height,
                size_tolerance=0.3,
                lang_threshold=0.2,
                skip_ocr=skip_ocr
            )

            if validation["valid"]:
                valid_chunks.append({
                    "index": chunk_index,
                    "key": chunk_key,
                    "base64": chunk_base64,
                    "reason": validation["reason"]
                })
            else:
                defective_chunks.append({
                    "index": chunk_index,
                    "key": chunk_key,
                    "reason": validation["reason"],
                    "defect_type": validation.get("defect_type", "unknown"),
                    "can_retry": validation.get("can_retry", True),
                    "original_width": expected_width,
                    "original_height": expected_height
                })

            # 메모리 정리: 1청크 처리 → 분류 → 즉시 캐시 삭제
            chunk["base64"] = None
            chunk_base64 = None
            del validation
            gc.collect()
            logger.info(f"[ValidateImageChunks] chunk {chunk_index} done, gc.collect() completed")

        all_valid = len(defective_chunks) == 0

        logger.info(f"[ValidateImageChunks] {image_key}: {len(valid_chunks)} valid, {len(defective_chunks)} defective")

        return jsonify({
            "success": True,
            "image_key": image_key,
            "all_valid": all_valid,
            "valid_chunks": valid_chunks,
            "defective_chunks": defective_chunks,
            "stats": {
                "total": len(chunks),
                "valid": len(valid_chunks),
                "defective": len(defective_chunks)
            }
        })

    except Exception as e:
        logger.error(f"[ValidateImageChunks] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/merge-and-save-image", methods=["POST"])
def merge_and_save_image():
    """
    청크들을 병합하고 Supabase Storage에 저장 (n8n 루프용)

    정상 청크들을 병합하여 최종 이미지 생성 후 저장

    Request:
        image_key: 이미지 식별자
        product_id: 상품 ID
        image_index: 이미지 인덱스
        chunks: 청크 리스트 [{index, base64}, ...]
        config: {
            supabase_url, supabase_key, storage_bucket,
            output_format, output_quality, output_extension
        }
        batch_job_id: 배치 작업 ID (선택)

    Response:
        success: 성공 여부
        merged_url: 병합된 이미지 URL
    """
    try:
        data = request.get_json()

        image_key = data.get("image_key")
        product_id = data.get("product_id")
        image_index = data.get("image_index")
        chunks = data.get("chunks", [])
        config = data.get("config", {})
        batch_job_id = data.get("batch_job_id")

        if not chunks:
            return jsonify({"error": "chunks required"}), 400

        logger.info(f"[MergeAndSave] Merging {len(chunks)} chunks for {image_key}")

        # 설정
        supabase_url = config.get("supabase_url") or SUPABASE_URL
        supabase_key = config.get("supabase_key") or SUPABASE_SERVICE_KEY
        storage_bucket = config.get("storage_bucket", "translated-images")
        output_format = config.get("output_format", "WEBP").upper()
        output_quality = int(config.get("output_quality", 100))
        output_extension = config.get("output_extension", ".webp")

        # 청크 정렬
        sorted_chunks = sorted(chunks, key=lambda x: x.get("index", 0))

        # 원본 너비 추정 (첫 번째 청크 기준)
        original_width = None
        if sorted_chunks:
            first_chunk = sorted_chunks[0].get("base64")
            if first_chunk:
                try:
                    img_bytes = base64.b64decode(first_chunk)
                    img = Image.open(BytesIO(img_bytes))
                    original_width = img.width
                except:
                    pass

        # 병합
        merge_result = merge_images_internal(
            sorted_chunks,
            original_width or 800,
            output_format=output_format,
            output_quality=output_quality
        )

        if not merge_result["success"]:
            return jsonify({
                "success": False,
                "error": f"병합 실패: {merge_result['error']}"
            }), 400

        # Supabase Storage에 업로드
        storage_path = f"{product_id}/{image_index}{output_extension}"

        mime_types = {
            "WEBP": "image/webp",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png"
        }
        mime_type = mime_types.get(output_format, "image/webp")

        upload_result = upload_to_supabase_storage(
            merge_result["merged_base64"],
            storage_path,
            storage_bucket,
            supabase_url,
            supabase_key,
            content_type=mime_type
        )

        if not upload_result["success"]:
            return jsonify({
                "success": False,
                "error": f"업로드 실패: {upload_result['error']}"
            }), 400

        logger.info(f"[MergeAndSave] Saved to: {upload_result['url']}")

        return jsonify({
            "success": True,
            "image_key": image_key,
            "product_id": product_id,
            "image_index": image_index,
            "merged_url": upload_result["url"],
            "dimensions": {
                "width": merge_result.get("width"),
                "height": merge_result.get("height")
            }
        })

    except Exception as e:
        logger.error(f"[MergeAndSave] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/record-defective-chunks", methods=["POST"])
def record_defective_chunks():
    """
    불량 청크 + 정상 청크를 DB에 기록 (n8n 루프용)

    불량이 있는 이미지의 모든 청크를 DB에 저장
    - 불량 청크: status='invalid', 재처리 대상
    - 정상 청크: status='valid', translated_base64 보관

    재처리 완료 후 정상 청크 + 재처리된 청크를 합쳐서 병합

    Request:
        batch_job_id: 배치 작업 ID
        image_key: 이미지 식별자
        product_id: 상품 ID
        image_index: 이미지 인덱스
        defective_chunks: [{index, key, reason, defect_type, can_retry}, ...]
        valid_chunks: [{index, key, base64}, ...]  # 정상 청크 (보관용)
        total_chunks: 전체 청크 수
        storage_batch_id: 원본 청크 저장 배치 ID (Storage 경로용)
        config: {supabase_url, supabase_key}

    Response:
        recorded_count: 기록된 청크 수
        valid_count: 정상 청크 수
        invalid_count: 불량 청크 수
    """
    try:
        data = request.get_json()

        batch_job_id = data.get("batch_job_id")
        image_key = data.get("image_key")
        product_id = data.get("product_id")
        image_index = data.get("image_index")
        defective_chunks = data.get("defective_chunks", [])
        valid_chunks = data.get("valid_chunks", [])
        total_chunks = data.get("total_chunks", len(defective_chunks) + len(valid_chunks))
        storage_batch_id = data.get("storage_batch_id")
        config = data.get("config", {})

        if not batch_job_id:
            return jsonify({"error": "batch_job_id required"}), 400

        if not defective_chunks and not valid_chunks:
            return jsonify({"error": "defective_chunks or valid_chunks required"}), 400

        supabase_url = config.get("supabase_url") or SUPABASE_URL
        supabase_key = config.get("supabase_key") or SUPABASE_SERVICE_KEY

        logger.info(f"[RecordChunks] Recording {len(defective_chunks)} defective + {len(valid_chunks)} valid chunks for {image_key}")

        # 1. image_processing 레코드 확인/생성
        success, ip_result = supabase_request(
            "GET",
            f"image_processing?batch_job_id=eq.{batch_job_id}&product_id=eq.{product_id}&image_index=eq.{image_index}",
            supabase_url, supabase_key
        )

        if success and ip_result:
            image_processing_id = ip_result[0]["id"]
            # 상태 업데이트
            supabase_request(
                "PATCH",
                f"image_processing?id=eq.{image_processing_id}",
                supabase_url, supabase_key,
                {
                    "status": "partial",
                    "total_chunks": total_chunks,
                    "valid_chunks": len(valid_chunks),
                    "invalid_chunks": len(defective_chunks)
                }
            )
        else:
            # 새로 생성
            success, ip_result = supabase_request(
                "POST",
                "image_processing",
                supabase_url, supabase_key,
                {
                    "batch_job_id": batch_job_id,
                    "product_id": product_id,
                    "image_index": image_index,
                    "status": "partial",
                    "total_chunks": total_chunks,
                    "valid_chunks": len(valid_chunks),
                    "invalid_chunks": len(defective_chunks)
                }
            )
            if not success:
                return jsonify({"error": f"Failed to create image_processing: {ip_result}"}), 500
            image_processing_id = ip_result[0]["id"] if isinstance(ip_result, list) else ip_result.get("id")

        recorded_count = 0

        # 2. 불량 청크 기록
        for chunk in defective_chunks:
            chunk_index = chunk.get("index")
            chunk_key = chunk.get("key")

            # 기존 레코드 확인
            success, existing = supabase_request(
                "GET",
                f"chunk_processing?image_processing_id=eq.{image_processing_id}&chunk_index=eq.{chunk_index}",
                supabase_url, supabase_key
            )

            # 원본 청크 가져오기 (Storage에서)
            original_base64 = ""
            if storage_batch_id and chunk_key:
                orig_result = get_original_chunk(chunk_key, storage_batch_id, supabase_url, supabase_key)
                if orig_result["success"]:
                    original_base64 = orig_result["base64"]

            chunk_data = {
                "status": "invalid",
                "validation_result": {
                    "valid": False,
                    "reason": chunk.get("reason"),
                    "defect_type": chunk.get("defect_type")
                },
                "last_error": chunk.get("reason")
            }

            if existing:
                chunk_id = existing[0]["id"]
                supabase_request(
                    "PATCH",
                    f"chunk_processing?id=eq.{chunk_id}",
                    supabase_url, supabase_key,
                    chunk_data
                )
            else:
                chunk_data.update({
                    "image_processing_id": image_processing_id,
                    "chunk_index": chunk_index,
                    "original_base64": original_base64,
                    "retry_count": 0
                })
                supabase_request(
                    "POST",
                    "chunk_processing",
                    supabase_url, supabase_key,
                    chunk_data
                )

            recorded_count += 1

        # 3. 정상 청크도 기록 (translated_base64 보관)
        for chunk in valid_chunks:
            chunk_index = chunk.get("index")
            chunk_key = chunk.get("key")
            translated_base64 = chunk.get("base64", "")

            # 기존 레코드 확인
            success, existing = supabase_request(
                "GET",
                f"chunk_processing?image_processing_id=eq.{image_processing_id}&chunk_index=eq.{chunk_index}",
                supabase_url, supabase_key
            )

            chunk_data = {
                "status": "valid",
                "translated_base64": translated_base64,
                "validation_result": {
                    "valid": True,
                    "reason": chunk.get("reason", "검증 통과")
                }
            }

            if existing:
                chunk_id = existing[0]["id"]
                supabase_request(
                    "PATCH",
                    f"chunk_processing?id=eq.{chunk_id}",
                    supabase_url, supabase_key,
                    chunk_data
                )
            else:
                chunk_data.update({
                    "image_processing_id": image_processing_id,
                    "chunk_index": chunk_index,
                    "retry_count": 0
                })
                supabase_request(
                    "POST",
                    "chunk_processing",
                    supabase_url, supabase_key,
                    chunk_data
                )

            recorded_count += 1

        logger.info(f"[RecordChunks] Recorded {recorded_count} chunks (valid: {len(valid_chunks)}, invalid: {len(defective_chunks)})")

        return jsonify({
            "success": True,
            "image_key": image_key,
            "recorded_count": recorded_count,
            "valid_count": len(valid_chunks),
            "invalid_count": len(defective_chunks),
            "image_processing_id": image_processing_id
        })

    except Exception as e:
        logger.error(f"[RecordChunks] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get-all-defective-chunks", methods=["POST"])
def get_all_defective_chunks():
    """
    배치 작업의 모든 불량 청크 조회 (재처리 배치 생성용)

    모든 이미지 처리 완료 후 불량 청크 일괄 조회하여
    재처리 배치 생성에 사용

    Request:
        batch_job_id: 배치 작업 ID
        config: {supabase_url, supabase_key}

    Response:
        defective_chunks: [{chunk_id, chunk_key, original_base64, ...}, ...]
        total_count: 총 불량 청크 수
    """
    try:
        data = request.get_json()

        batch_job_id = data.get("batch_job_id")
        config = data.get("config", {})

        if not batch_job_id:
            return jsonify({"error": "batch_job_id required"}), 400

        supabase_url = config.get("supabase_url") or SUPABASE_URL
        supabase_key = config.get("supabase_key") or SUPABASE_SERVICE_KEY

        logger.info(f"[GetDefectiveChunks] Fetching for batch {batch_job_id}")

        # image_processing IDs 조회
        success, ip_result = supabase_request(
            "GET",
            f"image_processing?batch_job_id=eq.{batch_job_id}&select=id,product_id,image_index",
            supabase_url, supabase_key
        )

        if not success or not ip_result:
            return jsonify({
                "success": True,
                "defective_chunks": [],
                "total_count": 0
            })

        image_ids = [r["id"] for r in ip_result]

        # chunk_processing에서 불량 청크 조회 (invalid/pending/retrying 모두 포함)
        # retry_count 필터 제거 — create-retry-batch에서 retry_count >= 3이면 replaced 처리
        success, chunks = supabase_request(
            "GET",
            f"chunk_processing?image_processing_id=in.({','.join(map(str, image_ids))})&status=in.(invalid,pending,retrying)&select=*",
            supabase_url, supabase_key
        )

        if not success:
            return jsonify({"error": f"Failed to fetch chunks: {chunks}"}), 500

        # image_processing 매핑
        ip_map = {r["id"]: r for r in ip_result}

        defective_chunks = []
        for chunk in chunks or []:
            ip_id = chunk.get("image_processing_id")
            ip_data = ip_map.get(ip_id, {})

            defective_chunks.append({
                "chunk_id": chunk["id"],
                "chunk_index": chunk["chunk_index"],
                "product_id": ip_data.get("product_id"),
                "image_index": ip_data.get("image_index"),
                "original_base64": chunk.get("original_base64", ""),
                "retry_count": chunk.get("retry_count", 0),
                "validation_result": chunk.get("validation_result")
            })

        logger.info(f"[GetDefectiveChunks] Found {len(defective_chunks)} defective chunks")

        return jsonify({
            "success": True,
            "defective_chunks": defective_chunks,
            "total_count": len(defective_chunks)
        })

    except Exception as e:
        logger.error(f"[GetDefectiveChunks] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def merge_image_from_db_core(image_processing_id, config=None):
    """
    DB에 저장된 청크들을 가져와서 병합/저장 (내부 호출용)

    Args:
        image_processing_id: 이미지 처리 ID
        config: {supabase_url, supabase_key, storage_bucket, ...}

    Returns:
        dict: {success, merged_url, dimensions, ...} (plain dict, not jsonify)
    """
    try:
        if config is None:
            config = {}

        if not image_processing_id:
            return {"success": False, "error": "image_processing_id required"}

        supabase_url = config.get("supabase_url") or SUPABASE_URL
        supabase_key = config.get("supabase_key") or SUPABASE_SERVICE_KEY
        storage_bucket = config.get("storage_bucket", "translated-images")
        output_format = config.get("output_format", "WEBP").upper()
        output_quality = int(config.get("output_quality", 100))
        output_extension = config.get("output_extension", ".webp")

        logger.info(f"[MergeFromDB] Merging image {image_processing_id}")

        # 1. image_processing 정보 조회
        success, ip_result = supabase_request(
            "GET",
            f"image_processing?id=eq.{image_processing_id}&select=*",
            supabase_url, supabase_key
        )

        if not success or not ip_result:
            return {"success": False, "error": "image_processing not found"}

        ip_data = ip_result[0]
        product_id = ip_data.get("product_id")
        image_index = ip_data.get("image_index")

        # 2. 모든 청크 조회
        success, chunks = supabase_request(
            "GET",
            f"chunk_processing?image_processing_id=eq.{image_processing_id}&select=*&order=chunk_index",
            supabase_url, supabase_key
        )

        if not success or not chunks:
            return {"success": False, "error": "No chunks found"}

        # 3. 모든 청크가 valid인지 확인
        invalid_chunks = [c for c in chunks if c.get("status") != "valid"]
        if invalid_chunks:
            return {
                "success": False,
                "error": f"Not all chunks are valid. {len(invalid_chunks)} chunks still invalid.",
                "invalid_count": len(invalid_chunks)
            }

        # 4. 청크 데이터 준비
        chunk_list = []
        for chunk in chunks:
            translated_base64 = chunk.get("translated_base64")
            if not translated_base64:
                return {
                    "success": False,
                    "error": f"Chunk {chunk['chunk_index']} has no translated_base64"
                }

            chunk_list.append({
                "index": chunk["chunk_index"],
                "base64": translated_base64
            })

        # 5. 병합
        sorted_chunks = sorted(chunk_list, key=lambda x: x["index"])

        # 원본 너비 추정 (첫 번째 청크 기준)
        original_width = None
        if sorted_chunks:
            first_chunk = sorted_chunks[0].get("base64")
            if first_chunk:
                try:
                    img_bytes = base64.b64decode(first_chunk)
                    img = Image.open(BytesIO(img_bytes))
                    original_width = img.width
                except:
                    pass

        merge_result = merge_images_internal(
            sorted_chunks,
            original_width or 800,
            output_format=output_format,
            output_quality=output_quality
        )

        if not merge_result["success"]:
            return {
                "success": False,
                "error": f"병합 실패: {merge_result['error']}"
            }

        # 6. Supabase Storage에 업로드
        storage_path = f"{product_id}/{image_index}{output_extension}"

        mime_types = {
            "WEBP": "image/webp",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png"
        }
        mime_type = mime_types.get(output_format, "image/webp")

        upload_result = upload_to_supabase_storage(
            merge_result["merged_base64"],
            storage_path,
            storage_bucket,
            supabase_url,
            supabase_key,
            content_type=mime_type
        )

        if not upload_result["success"]:
            return {
                "success": False,
                "error": f"업로드 실패: {upload_result['error']}"
            }

        # 7. image_processing 상태 업데이트
        supabase_request(
            "PATCH",
            f"image_processing?id=eq.{image_processing_id}",
            supabase_url, supabase_key,
            {
                "status": "completed",
                "merged_url": upload_result["url"]
            }
        )

        # 8. 청크 데이터 정리 (메모리 절약)
        chunk_ids = [c["id"] for c in chunks]
        for chunk_id in chunk_ids:
            supabase_request(
                "PATCH",
                f"chunk_processing?id=eq.{chunk_id}",
                supabase_url, supabase_key,
                {
                    "translated_base64": None,
                    "original_base64": None
                }
            )

        logger.info(f"[MergeFromDB] Saved to: {upload_result['url']}, cleaned up {len(chunk_ids)} chunks")

        return {
            "success": True,
            "image_processing_id": image_processing_id,
            "product_id": product_id,
            "image_index": image_index,
            "merged_url": upload_result["url"],
            "chunks_cleaned": len(chunk_ids),
            "dimensions": {
                "width": merge_result.get("width"),
                "height": merge_result.get("height")
            }
        }

    except Exception as e:
        logger.error(f"[MergeFromDB] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.route("/process-retry-result", methods=["POST"])
def process_retry_result():
    """
    재처리 배치 결과 처리 (n8n 루프용)

    재처리된 청크를 검증하고 DB 업데이트
    모든 청크가 valid가 되면 병합/저장

    Request:
        image_key: 이미지 식별자
        retry_chunks: 재처리된 청크 리스트 [{index, base64, key}, ...]
        batch_job_id: 배치 작업 ID
        target_lang: 타겟 언어
        config: {supabase_url, supabase_key, storage_bucket, ...}

    Response:
        all_valid: 모든 청크가 정상인지
        merged_url: 병합된 이미지 URL (모든 청크 정상 시)
        still_invalid: 여전히 불량인 청크 수
    """
    try:
        data = request.get_json()

        image_key = data.get("image_key")
        retry_chunks = data.get("retry_chunks", [])
        batch_job_id = data.get("batch_job_id")
        target_lang = data.get("target_lang", "en")
        source_lang = data.get("source_lang")  # 원본 언어 (이중 검사용)
        config = data.get("config", {})

        if not image_key or not retry_chunks:
            return jsonify({"error": "image_key and retry_chunks required"}), 400

        supabase_url = config.get("supabase_url") or SUPABASE_URL
        supabase_key = config.get("supabase_key") or SUPABASE_SERVICE_KEY

        # image_key에서 product_id, image_index 추출
        parts = image_key.split("_")
        if len(parts) < 2:
            return jsonify({"error": "Invalid image_key format"}), 400

        product_id = int(parts[0])
        image_index = int(parts[1])

        logger.info(f"[ProcessRetryResult] Processing {len(retry_chunks)} retry chunks for {image_key}")

        # 1. image_processing 조회
        success, ip_result = supabase_request(
            "GET",
            f"image_processing?batch_job_id=eq.{batch_job_id}&product_id=eq.{product_id}&image_index=eq.{image_index}",
            supabase_url, supabase_key
        )

        if not success or not ip_result:
            return jsonify({"error": "image_processing not found"}), 404

        image_processing_id = ip_result[0]["id"]

        # 2. 재처리된 청크 검증 및 DB 업데이트
        import gc
        still_invalid = 0
        for i, chunk in enumerate(retry_chunks):
            chunk_index = chunk.get("index")
            chunk_base64 = chunk.get("base64")

            if not chunk_base64:
                still_invalid += 1
                continue

            # Gemini 출력 → 원본 크기로 리사이즈
            retry_expected_w = chunk.get("width")
            retry_expected_h = chunk.get("height")
            if retry_expected_w or retry_expected_h:
                chunk_base64, was_resized, orig_w, orig_h = resize_chunk_to_original(
                    chunk_base64, retry_expected_w, retry_expected_h
                )
                if was_resized:
                    logger.info(f"[ProcessRetryResult] chunk {chunk_index}: resized {orig_w}x{orig_h} → {retry_expected_w}x{retry_expected_h}")

            # 검증 (source_lang이 있으면 이중 검사)
            validation = validate_translated_chunk(
                chunk_base64,
                target_lang=target_lang,
                source_lang=source_lang,
                expected_width=retry_expected_w,
                expected_height=retry_expected_h,
                skip_ocr=False
            )

            # DB 업데이트
            if validation["valid"]:
                supabase_request(
                    "PATCH",
                    f"chunk_processing?image_processing_id=eq.{image_processing_id}&chunk_index=eq.{chunk_index}",
                    supabase_url, supabase_key,
                    {
                        "status": "valid",
                        "translated_base64": chunk_base64,
                        "validation_result": validation
                    }
                )
            else:
                still_invalid += 1
                supabase_request(
                    "PATCH",
                    f"chunk_processing?image_processing_id=eq.{image_processing_id}&chunk_index=eq.{chunk_index}",
                    supabase_url, supabase_key,
                    {
                        "status": "invalid",
                        "validation_result": validation,
                        "last_error": validation.get("reason")
                    }
                )

            # 메모리 정리: 1청크 처리 → 분류 → 즉시 캐시 삭제
            chunk["base64"] = None
            chunk_base64 = None
            del validation
            gc.collect()

        # 3. 모든 청크가 valid인지 확인
        success, all_chunks = supabase_request(
            "GET",
            f"chunk_processing?image_processing_id=eq.{image_processing_id}&select=status",
            supabase_url, supabase_key
        )

        invalid_count = len([c for c in (all_chunks or []) if c.get("status") != "valid"])
        all_valid = invalid_count == 0

        result = {
            "success": True,
            "image_key": image_key,
            "all_valid": all_valid,
            "still_invalid": invalid_count
        }

        # 4. 모든 청크가 정상이면 병합/저장
        if all_valid:
            merge_result = merge_image_from_db_core(image_processing_id, config)
            if merge_result.get("success"):
                result["merged_url"] = merge_result.get("merged_url")
                result["dimensions"] = merge_result.get("dimensions")
            else:
                result["merge_error"] = merge_result.get("error")

        logger.info(f"[ProcessRetryResult] {image_key}: all_valid={all_valid}, still_invalid={invalid_count}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ProcessRetryResult] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting Text Render Service v10.2 on port {port}")
    logger.info(f"Features: slice, merge, batch-results-v2, retry-batch, n8n-loop-api")
    logger.info(f"n8n Loop APIs: get-batch-images, validate-image-chunks, merge-and-save-image, record-defective-chunks, get-all-defective-chunks, process-retry-result")
    logger.info(f"Vertex AI available: {vertex_ai_available}")
    app.run(host="0.0.0.0", port=port, debug=True)
