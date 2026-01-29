"""
텍스트 번역 및 렌더링 서비스 v4
- OpenCV 기반 자연스러운 인페인팅
- 한글→영어 길이 차이 고려한 폰트 크기 조정
- 텍스트 줄바꿈 및 자동 맞춤
- 원본 스타일 정확한 매칭
- 긴 이미지 슬라이스/병합 지원 (v4 신규)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import requests
from io import BytesIO
import base64
import os
import logging
import json
from collections import Counter
from functools import lru_cache
import re

app = Flask(__name__)

# ===== 성능 최적화 설정 =====
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_DIMENSION = 30000  # 30000px
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Cloud 설정 (선택사항 - 슬라이스/병합에는 불필요)
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Supabase 설정 (환경변수에서 가져옴 - 배치 처리용)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# 폰트 경로 설정
FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]

BOLD_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]

# Vertex AI 초기화 시도
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


def _find_available_font_path(bold=False):
    """사용 가능한 폰트 경로 찾기 (캐시용)"""
    paths = BOLD_FONT_PATHS if bold else FONT_PATHS
    for font_path in paths:
        if os.path.exists(font_path):
            return font_path
    return None

# 폰트 경로 캐시 (시작 시 한 번만 검색)
_CACHED_FONT_PATH = _find_available_font_path(bold=False)
_CACHED_BOLD_FONT_PATH = _find_available_font_path(bold=True)


@lru_cache(maxsize=64)
def get_font(size=24, bold=False):
    """시스템에서 사용 가능한 폰트 찾기 (LRU 캐시 적용)"""
    font_path = _CACHED_BOLD_FONT_PATH if bold else _CACHED_FONT_PATH
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def download_image(url):
    """URL에서 이미지 다운로드"""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


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


def cv2_to_pil(cv2_image):
    """OpenCV 이미지를 PIL Image로 변환"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def get_bounds_rect(bounds):
    """bounds에서 x, y, width, height 추출"""
    vertices = bounds.get("vertices", [])
    if vertices and len(vertices) >= 4:
        xs = [v.get("x", 0) or 0 for v in vertices]
        ys = [v.get("y", 0) or 0 for v in vertices]
        x = min(xs)
        y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        w = max(1, int(max_x - x))  # 최소 1 보장
        h = max(1, int(max_y - y))  # 최소 1 보장
        return int(x), int(y), w, h
    else:
        return (
            int(bounds.get("x", 0) or 0),
            int(bounds.get("y", 0) or 0),
            max(1, int(bounds.get("width", 100) or 100)),
            max(1, int(bounds.get("height", 30) or 30))
        )


def create_mask_for_inpainting(image_size, text_blocks, padding=5):
    """인페인팅용 마스크 생성 (OpenCV 형식)"""
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    for block in text_blocks:
        bounds = block.get("bounds", {})
        x, y, w, h = get_bounds_rect(bounds)

        # 패딩 적용
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_size[0], x + w + padding)
        y2 = min(image_size[1], y + h + padding)

        # 흰색으로 마스크 영역 표시
        mask[y1:y2, x1:x2] = 255

    return mask


def opencv_inpaint(image, text_blocks, padding=8):
    """
    OpenCV 기반 자연스러운 인페인팅
    - Telea 알고리즘과 Navier-Stokes 알고리즘 중 더 나은 결과 선택
    """
    cv_image = pil_to_cv2(image)
    mask = create_mask_for_inpainting(image.size, text_blocks, padding)

    # 마스크 약간 확장 (경계 처리 개선)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 두 가지 인페인팅 알고리즘 시도
    # INPAINT_TELEA: Fast Marching Method - 빠르고 일반적으로 좋은 결과
    # INPAINT_NS: Navier-Stokes - 더 부드러운 결과

    inpaint_radius = 5  # 인페인팅 반경

    # Telea 알고리즘 사용 (일반적으로 더 좋은 결과)
    result = cv2.inpaint(cv_image, mask, inpaint_radius, cv2.INPAINT_TELEA)

    return cv2_to_pil(result)


def advanced_inpaint(image, text_blocks, padding=10):
    """
    고급 인페인팅: 주변 색상/패턴 분석 후 자연스럽게 채우기
    """
    cv_image = pil_to_cv2(image)
    result = cv_image.copy()

    for block in text_blocks:
        try:
            bounds = block.get("bounds", {})
            x, y, w, h = get_bounds_rect(bounds)

            # 유효성 검사
            if w <= 0 or h <= 0:
                continue

            # 영역 확장
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.size[0], x + w + padding)
            y2 = min(image.size[1], y + h + padding)

            # 영역이 유효한지 확인
            if x2 <= x1 or y2 <= y1:
                continue

            # 주변 영역 샘플링 (위, 아래, 좌, 우)
            sample_size = 15
            samples = []

            # 위쪽
            if y1 >= sample_size:
                samples.append(cv_image[y1-sample_size:y1, x1:x2])
            # 아래쪽
            if y2 + sample_size <= image.size[1]:
                samples.append(cv_image[y2:y2+sample_size, x1:x2])
            # 왼쪽
            if x1 >= sample_size:
                samples.append(cv_image[y1:y2, x1-sample_size:x1])
            # 오른쪽
            if x2 + sample_size <= image.size[0]:
                samples.append(cv_image[y1:y2, x2:x2+sample_size])

            if samples:
                # 주변 색상의 평균과 표준편차 계산
                valid_samples = [s for s in samples if s.size > 0]
                if valid_samples:
                    all_pixels = np.vstack([s.reshape(-1, 3) for s in valid_samples])
                    if len(all_pixels) > 0:
                        mean_color = np.mean(all_pixels, axis=0).astype(np.uint8)

                        # 그라데이션 효과로 자연스럽게 채우기
                        fill_region = result[y1:y2, x1:x2]

                        # 가우시안 블러된 배경색으로 채우기
                        if fill_region.size > 0:
                            fill_region[:] = mean_color

                            # 경계 부드럽게 처리
                            region_h = y2 - y1
                            region_w = x2 - x1
                            if region_h > padding * 2 and region_w > padding * 2:
                                mask_region = np.zeros((region_h, region_w), dtype=np.uint8)
                                mask_region[padding:-padding, padding:-padding] = 255

                                if mask_region.shape[0] > 0 and mask_region.shape[1] > 0:
                                    mask_region = cv2.GaussianBlur(mask_region, (21, 21), 0)

                                    # 원본과 블렌딩
                                    mask_normalized = mask_region.astype(np.float32) / 255.0
                                    for c in range(3):
                                        fill_region[:, :, c] = (
                                            fill_region[:, :, c] * mask_normalized +
                                            cv_image[y1:y2, x1:x2, c] * (1 - mask_normalized)
                                        ).astype(np.uint8)

        except Exception as block_err:
            logger.warning(f"Block inpaint error: {block_err}")

    # 최종 OpenCV 인페인팅으로 마무리
    mask = create_mask_for_inpainting(image.size, text_blocks, padding=3)
    result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)

    return cv2_to_pil(result)


def extract_text_style(image, bounds):
    """텍스트 영역에서 원본 스타일 정확하게 추출"""
    style = {
        "color": (0, 0, 0),
        "font_size": 24,
        "bold": False,
        "background_color": (255, 255, 255)
    }

    try:
        x, y, w, h = get_bounds_rect(bounds)

        if w <= 0 or h <= 0:
            return style

        # 이미지 범위 체크
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.width - x)
        h = min(h, image.height - y)

        if w <= 0 or h <= 0:
            return style

        # 텍스트 영역 크롭
        region = image.crop((x, y, x + w, y + h))
        pixels = list(region.getdata())

        if not pixels:
            return style

        # K-means 클러스터링으로 주요 색상 추출 (배경 vs 텍스트)
        pixel_array = np.array(pixels, dtype=np.float32)

        # 2개의 클러스터로 분류 (배경, 텍스트)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2

        try:
            _, labels, centers = cv2.kmeans(pixel_array, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # 각 클러스터의 픽셀 수
            unique, counts = np.unique(labels, return_counts=True)

            # 더 많은 픽셀 = 배경, 적은 픽셀 = 텍스트
            if len(counts) >= 2:
                bg_idx = np.argmax(counts)
                text_idx = np.argmin(counts)

                bg_color = tuple(centers[bg_idx].astype(int))
                text_color = tuple(centers[text_idx].astype(int))

                # 명도 차이 확인
                bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]

                if abs(bg_brightness - text_brightness) > 20:
                    style["color"] = text_color
                    style["background_color"] = bg_color
                else:
                    # 차이가 적으면 밝기 기반
                    if bg_brightness > 128:
                        style["color"] = (0, 0, 0)
                    else:
                        style["color"] = (255, 255, 255)
                    style["background_color"] = bg_color
        except:
            # K-means 실패 시 간단한 방법 사용
            color_counts = Counter(pixels)
            most_common = color_counts.most_common(2)
            if len(most_common) >= 2:
                style["background_color"] = most_common[0][0][:3]
                style["color"] = most_common[1][0][:3]

        # 폰트 크기: 높이의 75-85%
        style["font_size"] = max(10, int(h * 0.8))

        # 굵기 추정: 텍스트 픽셀 비율
        text_color = style["color"]
        if pixels and len(pixels) > 0:
            similar_count = sum(
                1 for p in pixels
                if len(p) >= 3
                and abs(p[0] - text_color[0]) < 60
                and abs(p[1] - text_color[1]) < 60
                and abs(p[2] - text_color[2]) < 60
            )
            text_ratio = similar_count / len(pixels)
            style["bold"] = text_ratio > 0.35

    except Exception as e:
        logger.warning(f"Style extraction failed: {e}")

    return style


def estimate_text_length_ratio(korean_text, english_text):
    """한글과 영어 텍스트의 렌더링 길이 비율 추정"""
    # 한글은 보통 영어보다 짧게 렌더링됨 (같은 의미일 때 영어가 더 길다)
    # 일반적으로 영어는 한글의 1.3~1.8배 길이

    if not korean_text or not english_text:
        return 1.0

    korean_chars = len(re.findall(r'[가-힣]', korean_text))
    english_chars = len(english_text)

    if korean_chars == 0:
        return 1.0

    # 영어 문자 수 / 한글 문자 수의 비율
    divisor = korean_chars * 1.5
    if divisor == 0:
        return 1.0

    return max(0.8, min(2.5, english_chars / divisor))


def wrap_text(text, font, max_width):
    """텍스트를 최대 너비에 맞게 줄바꿈"""
    if not text:
        return [""]

    # max_width가 너무 작으면 최소값 보장
    max_width = max(50, max_width)

    words = text.split()
    if not words:
        return [text]

    lines = []
    current_line = []

    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    for word in words:
        test_line = ' '.join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
        except:
            width = len(test_line) * 10  # fallback

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines if lines else [text]


def calculate_optimal_font_size(text, target_width, target_height, bold=False, allow_wrap=True):
    """
    텍스트에 최적화된 폰트 크기 계산
    - 영역에 맞추면서 가독성 유지
    - 필요시 줄바꿈 허용
    """
    # 0 방지
    target_width = max(10, target_width)
    target_height = max(10, target_height)

    max_size = min(72, max(12, int(target_height * 0.9)))
    min_size = max(8, int(target_height * 0.3))

    best_font = None
    best_lines = [text]
    best_size = min_size

    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    for size in range(max_size, min_size - 1, -1):
        font = get_font(size, bold)

        # 단일 라인 시도
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width <= target_width * 0.95 and text_height <= target_height * 0.95:
            return font, [text], text_width, text_height

        # 줄바꿈 시도
        if allow_wrap and size >= min_size + 4:
            lines = wrap_text(text, font, int(target_width * 0.95))

            if len(lines) <= 3:  # 최대 3줄까지
                total_height = 0
                max_line_width = 0

                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                    total_height += line_height * 1.2  # 줄 간격
                    max_line_width = max(max_line_width, line_width)

                if max_line_width <= target_width * 0.95 and total_height <= target_height * 0.95:
                    return font, lines, max_line_width, total_height

        best_font = font
        best_size = size

    # 최소 크기로 줄바꿈
    font = get_font(min_size, bold)
    lines = wrap_text(text, font, int(target_width * 0.95))

    return font, lines[:3], target_width, target_height


def render_text_block(draw, text, x, y, width, height, style, original_text=""):
    """단일 텍스트 블록 렌더링"""
    if not text:
        return

    # 0 방지
    width = max(10, width)
    height = max(10, height)

    text_color = style.get("color", (0, 0, 0))
    bold = style.get("bold", False)
    original_size = max(8, style.get("font_size", 24))

    # 한글→영어 길이 비율 고려
    if original_text:
        length_ratio = estimate_text_length_ratio(original_text, text)
        # 영어가 더 길면 폰트 크기를 약간 줄임
        if length_ratio > 1.2:
            original_size = int(original_size / (length_ratio * 0.7))

    # 최적 폰트 크기 계산
    font, lines, text_width, text_height = calculate_optimal_font_size(
        text, width, height, bold, allow_wrap=True
    )

    # 원본 크기와 비교하여 조정
    current_size = font.size if hasattr(font, 'size') else original_size
    if current_size > original_size * 1.3:
        font = get_font(int(original_size * 1.1), bold)
        lines = wrap_text(text, font, int(width * 0.95))

    # 텍스트 렌더링
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    total_height = 0
    line_heights = []
    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        total_height += line_height

    # 줄 간격 추가
    line_spacing = 4
    total_height += line_spacing * (len(lines) - 1)

    # 시작 Y 좌표 (수직 중앙 정렬)
    if height > total_height:
        current_y = y + (height - total_height) // 2
    else:
        current_y = y

    # 그림자 색상
    bg_brightness = sum(style.get("background_color", (255, 255, 255))) / 3
    text_brightness = sum(text_color) / 3

    if abs(bg_brightness - text_brightness) < 100:
        # 배경과 텍스트 대비가 낮으면 반대 색상 사용
        if bg_brightness > 128:
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)

    shadow_color = (255, 255, 255) if sum(text_color) < 384 else (0, 0, 0)

    for i, line in enumerate(lines):
        if not line:
            continue

        try:
            bbox = dummy_draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
        except:
            line_width = len(line) * 10

        # 수평 중앙 정렬
        if width > line_width:
            text_x = x + (width - line_width) // 2
        else:
            text_x = x

        # 그림자 (미세한 아웃라인 효과)
        for offset in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            draw.text(
                (text_x + offset[0], current_y + offset[1]),
                line,
                font=font,
                fill=shadow_color
            )

        # 메인 텍스트
        draw.text((text_x, current_y), line, font=font, fill=text_color)

        if i < len(line_heights):
            current_y += line_heights[i] + line_spacing
        else:
            current_y += 20 + line_spacing


def render_text_on_image(image, text_blocks, original_image=None):
    """이미지에 번역된 텍스트 렌더링 (원본 스타일 정확 매칭)"""
    draw = ImageDraw.Draw(image)
    style_source = original_image if original_image else image

    for block in text_blocks:
        translated_text = block.get("translatedText", "")
        original_text = block.get("text", "")
        bounds = block.get("bounds", {})

        if not translated_text:
            continue

        x, y, width, height = get_bounds_rect(bounds)

        if width <= 0 or height <= 0:
            continue

        # 원본 이미지에서 스타일 추출
        style = extract_text_style(style_source, bounds)

        # 텍스트 렌더링
        render_text_block(draw, translated_text, x, y, width, height, style, original_text)

        logger.info(f"Rendered: '{translated_text[:30]}...' at ({x}, {y}) size=({width}x{height})")

    return image


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "text-render-service-v8",
        "vertex_ai_available": vertex_ai_available,
        "opencv_available": True,
        "project_id": PROJECT_ID,
        "features": ["inpainting", "text-render", "slice", "merge"]
    })


@app.route("/render-text", methods=["POST"])
def render_text():
    """기본 텍스트 렌더링"""
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        image_base64_input = data.get("image_base64")
        text_blocks = data.get("text_blocks", [])

        if not image_url and not image_base64_input:
            return jsonify({"error": "image_url or image_base64 required"}), 400

        if image_url:
            image = download_image(image_url)
        else:
            image_data = base64.b64decode(image_base64_input)
            image = Image.open(BytesIO(image_data)).convert("RGB")

        result_image = render_text_on_image(image, text_blocks)
        result_base64, _ = image_to_base64(result_image)

        return jsonify({
            "success": True,
            "image_base64": result_base64,
            "text_blocks_rendered": len(text_blocks)
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/translate-image", methods=["POST"])
def translate_image():
    """
    전체 번역 파이프라인 (OpenCV 인페인팅 + 스타일 매칭 텍스트 삽입)
    """
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        image_base64_input = data.get("image_base64")
        text_blocks = data.get("text_blocks", [])
        use_advanced_inpaint = data.get("use_advanced_inpaint", True)

        if not image_url and not image_base64_input:
            return jsonify({"error": "image_url or image_base64 required"}), 400

        # 1. 이미지 로드
        logger.info("Step 1: Loading image...")
        try:
            if image_url:
                image = download_image(image_url)
            else:
                image_data = base64.b64decode(image_base64_input)
                image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as img_err:
            logger.error(f"Image load error: {img_err}")
            return jsonify({"error": f"Image load failed: {str(img_err)}"}), 400

        # 원본 이미지 저장 (스타일 추출용)
        original_image = image.copy()

        if not text_blocks:
            img_base64, _ = image_to_base64(image)
            return jsonify({
                "success": True,
                "image_base64": img_base64,
                "message": "No text blocks to process"
            })

        # text_blocks 유효성 검사
        valid_blocks = []
        for i, block in enumerate(text_blocks):
            try:
                bounds = block.get("bounds", {})
                x, y, w, h = get_bounds_rect(bounds)
                if w > 0 and h > 0 and x >= 0 and y >= 0:
                    valid_blocks.append(block)
                else:
                    logger.warning(f"Block {i} skipped: invalid bounds ({x}, {y}, {w}, {h})")
            except Exception as block_err:
                logger.warning(f"Block {i} skipped: {block_err}")

        if not valid_blocks:
            img_base64, _ = image_to_base64(image)
            return jsonify({
                "success": True,
                "image_base64": img_base64,
                "message": "No valid text blocks to process"
            })

        text_blocks = valid_blocks
        logger.info(f"Processing {len(text_blocks)} valid text blocks")

        # 2. OpenCV 인페인팅으로 텍스트 영역 제거
        logger.info(f"Step 2: Inpainting {len(text_blocks)} text regions...")

        try:
            if use_advanced_inpaint:
                inpainted_image = advanced_inpaint(image, text_blocks, padding=12)
            else:
                inpainted_image = opencv_inpaint(image, text_blocks, padding=10)
        except Exception as inpaint_err:
            logger.error(f"Inpainting error: {inpaint_err}")
            import traceback
            traceback.print_exc()
            # 인페인팅 실패 시 원본 이미지 사용
            inpainted_image = image.copy()

        # 3. 번역된 텍스트 삽입 (원본 스타일 매칭)
        logger.info("Step 3: Rendering translated text with matched style...")
        try:
            final_image = render_text_on_image(inpainted_image, text_blocks, original_image)
        except Exception as render_err:
            logger.error(f"Render error: {render_err}")
            import traceback
            traceback.print_exc()
            final_image = inpainted_image

        # 4. 결과 반환
        result_base64, _ = image_to_base64(final_image)

        return jsonify({
            "success": True,
            "image_base64": result_base64,
            "text_blocks_processed": len(text_blocks),
            "inpainting_method": "advanced_opencv" if use_advanced_inpaint else "opencv"
        })

    except Exception as e:
        logger.error(f"Error in translate_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/simple-replace", methods=["POST"])
def simple_replace():
    """간단한 텍스트 교체 (인페인팅 없이)"""
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        image_base64_input = data.get("image_base64")
        text_blocks = data.get("text_blocks", [])

        if not image_url and not image_base64_input:
            return jsonify({"error": "image_url or image_base64 required"}), 400

        if image_url:
            image = download_image(image_url)
        else:
            image_data = base64.b64decode(image_base64_input)
            image = Image.open(BytesIO(image_data)).convert("RGB")

        original_image = image.copy()

        # 배경색으로 채우기
        draw = ImageDraw.Draw(image)
        for block in text_blocks:
            bounds = block.get("bounds", {})
            x, y, w, h = get_bounds_rect(bounds)
            style = extract_text_style(original_image, bounds)
            bg_color = style.get("background_color", (255, 255, 255))
            draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill=bg_color)

        # 텍스트 렌더링
        final_image = render_text_on_image(image, text_blocks, original_image)
        result_base64, _ = image_to_base64(final_image)

        return jsonify({
            "success": True,
            "image_base64": result_base64,
            "text_blocks_processed": len(text_blocks)
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# ===== 이미지 슬라이스/병합 기능 (v4) =====

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


@app.route("/extract-style", methods=["POST"])
def extract_style():
    """텍스트 영역의 스타일 정보 추출"""
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        image_base64_input = data.get("image_base64")
        text_blocks = data.get("text_blocks", [])

        if not image_url and not image_base64_input:
            return jsonify({"error": "image_url or image_base64 required"}), 400

        if image_url:
            image = download_image(image_url)
        else:
            image_data = base64.b64decode(image_base64_input)
            image = Image.open(BytesIO(image_data)).convert("RGB")

        styles = []
        for block in text_blocks:
            bounds = block.get("bounds", {})
            style = extract_text_style(image, bounds)
            styles.append({
                "text": block.get("text", ""),
                "style": {
                    "color": list(style["color"]),
                    "font_size": style["font_size"],
                    "bold": style["bold"],
                    "background_color": list(style["background_color"])
                }
            })

        return jsonify({
            "success": True,
            "styles": styles
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# ===== Gemini 배치 결과 처리 (v5) =====

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

            image_key = f"{meta['productId']}_{meta['imageIndex']}"

            if image_key not in image_chunks:
                image_chunks[image_key] = {
                    "productId": meta["productId"],
                    "productCode": meta.get("productCode"),
                    "imageIndex": meta["imageIndex"],
                    "totalChunks": meta.get("totalChunks", 1),
                    "chunks": []
                }

            image_chunks[image_key]["chunks"].append({
                "index": meta["chunkIndex"],
                "base64": translated_base64,
                "height": meta.get("chunkHeight", 2000)
            })

        logger.info(f"Grouped into {len(image_chunks)} images")

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

                # 청크가 1개면 병합 불필요, 하지만 포맷 변환은 필요
                if len(sorted_chunks) == 1:
                    # 단일 청크도 출력 포맷에 맞게 변환
                    chunk_image = base64_to_image(sorted_chunks[0]["base64"])
                    merged_base64, _ = image_to_base64(chunk_image, format=pil_format, quality=output_quality)
                else:
                    # 청크 병합 후 출력 포맷 적용
                    merged_image = merge_images(sorted_chunks, overlap=0, blend_height=50)
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
        gemini_model: 사용할 모델 (기본: gemini-2.0-flash-exp)
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
        gemini_model = data.get("gemini_model", "gemini-2.0-flash-exp")
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

        gemini_model = config.get("geminiModel", "gemini-2.0-flash-exp")
        prompt = config.get("prompt", "Translate the text in this image.")
        chunk_height = config.get("chunkHeight", 3000)
        limit = config.get("limit", 1000)
        text_service_url = config.get("textServiceUrl", "https://text-render-service-production.up.railway.app")

        logger.info("=" * 60)
        logger.info(f"[prepare-batch] 시작")
        logger.info(f"[prepare-batch] 테이블: {table_name}")
        logger.info(f"[prepare-batch] 모델: {gemini_model}")
        logger.info(f"[prepare-batch] 청크 높이: {chunk_height}")
        logger.info(f"[prepare-batch] 조회 제한: {limit}")

        # 1. Supabase에서 직접 상품 조회
        logger.info(f"[prepare-batch] Supabase에서 상품 조회 중...")
        supabase_headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }

        # 번역이 필요한 상품만 조회 (translated_image가 null인 것)
        products_response = requests.get(
            f"{supabase_url}/rest/v1/{table_name}",
            headers=supabase_headers,
            params={
                "select": "id,product_code,description_images",
                "translated_image": "is.null",
                "description_images": "not.is.null",
                "order": "id",
                "limit": limit
            },
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

        # 2. 임시 파일에 JSONL 스트리밍 (메모리 효율)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        chunk_metadata = []
        total_chunks = 0
        errors = []
        processed_images = 0

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

                    chunk_metadata.append({
                        "key": request_key,
                        "productId": img_req["productId"],
                        "productCode": img_req["productCode"],
                        "imageIndex": img_req["imageIndex"],
                        "chunkIndex": chunk_idx,
                        "totalChunks": len(chunks),
                        "chunkHeight": chunk["height"],
                        "yStart": chunk["y_start"],
                        "yEnd": chunk["y_end"]
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
        batch_display_name = f"batch-{table_name}-{int(time.time())}"

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
            "batchName": batch_name,
            "batchState": batch_state,
            "batchDisplayName": batch_display_name,
            "fileName": file_name,
            "totalProducts": len(product_map),
            "totalImages": len(image_requests),
            "totalChunks": total_chunks,
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
                "outputQuality": config.get("outputQuality", 100)
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
    logger.info(f"Starting Text Render Service v8 on port {port}")
    logger.info(f"Features: inpainting, text-render, slice, merge, batch-results, translate-chunks, prepare-batch")
    logger.info(f"OpenCV inpainting: enabled")
    logger.info(f"Vertex AI available: {vertex_ai_available}")
    app.run(host="0.0.0.0", port=port, debug=True)
