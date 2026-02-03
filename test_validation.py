#!/usr/bin/env python3
"""
OCR 검증 시스템 E2E 테스트 (v2 - 이중 검사 포함)
- 다양한 케이스의 테스트 이미지 생성
- 검증 함수 직접 호출하여 결과 확인
"""

import sys
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 테스트 이미지 생성 함수들
def create_test_image(width, height, text, font_size=30, bg_color="white", text_color="black"):
    """테스트용 이미지 생성"""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 시스템 폰트 사용 시도
    try:
        # macOS
        font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", font_size)
    except:
        try:
            # Linux (Noto CJK)
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", font_size)
        except:
            font = ImageFont.load_default()

    # 텍스트 중앙 배치
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, font=font, fill=text_color)
    return img


def image_to_base64(img):
    """PIL Image를 base64로 변환"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_tests():
    """검증 테스트 실행"""
    print("=" * 60)
    print("OCR 검증 시스템 E2E 테스트 (v2 - 이중 검사)")
    print("=" * 60)

    # server.py에서 검증 함수 import
    try:
        from server import (
            validate_chunk,
            validate_translation_language,
            validate_translated_chunk,
            count_target_language_chars
        )
        print("[OK] 검증 함수 import 성공\n")
    except ImportError as e:
        print(f"[ERROR] Import 실패: {e}")
        print("서버 모듈 로드에 필요한 의존성을 확인하세요.")
        return False

    results = []

    # ========================================
    # 테스트 케이스 1: 정상 영어 텍스트
    # ========================================
    print("-" * 40)
    print("테스트 1: 정상 영어 텍스트 (기대: 통과)")
    print("-" * 40)

    img_en = create_test_image(400, 200, "Hello World\nThis is English text")
    b64_en = image_to_base64(img_en)

    result1 = validate_translated_chunk(
        b64_en,
        target_lang="en",
        source_lang="ko",  # 원본이 한국어였다고 가정
        expected_width=400,
        expected_height=200,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result1['valid']}")
    print(f"  - reason: {result1['reason']}")
    print(f"  - defect_type: {result1.get('defect_type')}")
    if result1.get('translation_validation'):
        tv = result1['translation_validation']
        print(f"  - source_lang_ratio: {tv.get('source_lang_ratio', 'N/A')}")
        print(f"  - total_chars: {tv.get('total_chars', 'N/A')}")

    results.append(("정상 영어 텍스트 (source_lang=ko)", result1['valid'] == True, result1))
    print()

    # ========================================
    # 테스트 케이스 2: 한국어 텍스트 (영어 타겟 - 실패해야 함) - 이중 검사
    # ========================================
    print("-" * 40)
    print("테스트 2: 한국어 텍스트, 영어 타겟, source_lang=ko (기대: 실패)")
    print("-" * 40)

    img_ko = create_test_image(400, 200, "안녕하세요\n한국어 텍스트입니다")
    b64_ko = image_to_base64(img_ko)

    result2 = validate_translated_chunk(
        b64_ko,
        target_lang="en",
        source_lang="ko",  # 원본 언어 지정 (이중 검사)
        expected_width=400,
        expected_height=200,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result2['valid']}")
    print(f"  - reason: {result2['reason']}")
    print(f"  - defect_type: {result2.get('defect_type')}")
    if result2.get('translation_validation'):
        tv = result2['translation_validation']
        print(f"  - source_lang_ratio: {tv.get('source_lang_ratio', 'N/A')}")
        print(f"  - total_chars: {tv.get('total_chars', 'N/A')}")

    results.append(("한국어→영어 미번역 (이중 검사)", result2['valid'] == False, result2))
    print()

    # ========================================
    # 테스트 케이스 3: 크기 불일치
    # ========================================
    print("-" * 40)
    print("테스트 3: 크기 불일치 (400x200 예상, 200x100 실제)")
    print("-" * 40)

    img_small = create_test_image(200, 100, "Small Image")
    b64_small = image_to_base64(img_small)

    result3 = validate_translated_chunk(
        b64_small,
        target_lang="en",
        expected_width=400,  # 예상 크기와 다름
        expected_height=200,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result3['valid']}")
    print(f"  - reason: {result3['reason']}")
    print(f"  - defect_type: {result3.get('defect_type')}")

    results.append(("크기 불일치", result3['valid'] == False and result3.get('defect_type') == 'size', result3))
    print()

    # ========================================
    # 테스트 케이스 4: 너무 작은 이미지 (50px 미만)
    # ========================================
    print("-" * 40)
    print("테스트 4: 너무 작은 이미지 (30x30)")
    print("-" * 40)

    img_tiny = create_test_image(30, 30, "X", font_size=15)
    b64_tiny = image_to_base64(img_tiny)

    result4 = validate_translated_chunk(
        b64_tiny,
        target_lang="en",
        expected_width=30,
        expected_height=30,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result4['valid']}")
    print(f"  - reason: {result4['reason']}")
    print(f"  - can_retry: {result4.get('can_retry')}")

    results.append(("너무 작은 이미지", result4['valid'] == False and result4.get('can_retry') == False, result4))
    print()

    # ========================================
    # 테스트 케이스 5: 텍스트 없는 이미지 (디코딩 기반 검증)
    # ========================================
    print("-" * 40)
    print("테스트 5: 텍스트 없는 이미지 (기대: 통과)")
    print("-" * 40)

    img_empty = Image.new("RGB", (400, 200), "white")
    b64_empty = image_to_base64(img_empty)

    print(f"  - Base64 길이: {len(b64_empty)} (이전 문제: 1000자 미만이면 실패)")

    result5 = validate_translated_chunk(
        b64_empty,
        target_lang="en",
        expected_width=400,
        expected_height=200,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result5['valid']}")
    print(f"  - reason: {result5['reason']}")
    if result5.get('translation_validation'):
        print(f"  - has_text: {result5['translation_validation'].get('has_text')}")

    results.append(("텍스트 없는 이미지 (디코딩 기반)", result5['valid'] == True, result5))
    print()

    # ========================================
    # 테스트 케이스 6: 일본어 텍스트, 일본어 타겟
    # ========================================
    print("-" * 40)
    print("테스트 6: 일본어 텍스트, 일본어 타겟 (기대: 통과)")
    print("-" * 40)

    img_ja = create_test_image(400, 200, "こんにちは\n日本語テキスト")
    b64_ja = image_to_base64(img_ja)

    result6 = validate_translated_chunk(
        b64_ja,
        target_lang="ja",
        source_lang="ko",  # 원본이 한국어였다고 가정
        expected_width=400,
        expected_height=200,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result6['valid']}")
    print(f"  - reason: {result6['reason']}")
    if result6.get('translation_validation'):
        tv = result6['translation_validation']
        print(f"  - source_lang_ratio: {tv.get('source_lang_ratio', 'N/A')}")

    results.append(("일본어 텍스트", result6['valid'] == True, result6))
    print()

    # ========================================
    # 테스트 케이스 7: count_target_language_chars 함수 테스트
    # ========================================
    print("-" * 40)
    print("테스트 7: count_target_language_chars 함수")
    print("-" * 40)

    test_cases = [
        ("Hello World", "en", 10),  # 10 영문자
        ("안녕하세요", "ko", 5),     # 5 한글
        ("こんにちは", "ja", 5),    # 5 히라가나
        ("混合テスト123", "ja", 5), # 混合テスト = 5자 (한자+카타카나)
    ]

    char_test_pass = True
    for text, lang, expected_min in test_cases:
        count = count_target_language_chars(text, lang)
        status = "OK" if count >= expected_min else "FAIL"
        if count < expected_min:
            char_test_pass = False
        print(f"  - '{text}' ({lang}): {count}자 (최소 {expected_min} 기대) [{status}]")

    results.append(("문자 카운트 함수", char_test_pass, None))
    print()

    # ========================================
    # 테스트 케이스 8: 중국어→영어 미번역 (이중 검사)
    # ========================================
    print("-" * 40)
    print("테스트 8: 중국어 텍스트, 영어 타겟, source_lang=zh (기대: 실패)")
    print("-" * 40)

    img_zh = create_test_image(400, 200, "你好世界\n中文文本测试")
    b64_zh = image_to_base64(img_zh)

    result8 = validate_translated_chunk(
        b64_zh,
        target_lang="en",
        source_lang="zh",  # 원본 언어: 중국어
        expected_width=400,
        expected_height=200,
        size_tolerance=0.3,
        lang_threshold=0.2
    )

    print(f"  - valid: {result8['valid']}")
    print(f"  - reason: {result8['reason']}")
    if result8.get('translation_validation'):
        tv = result8['translation_validation']
        print(f"  - source_lang_ratio: {tv.get('source_lang_ratio', 'N/A')}")
        print(f"  - total_chars: {tv.get('total_chars', 'N/A')}")

    results.append(("중국어→영어 미번역 (이중 검사)", result8['valid'] == False, result8))
    print()

    # ========================================
    # 결과 요약
    # ========================================
    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, passed_flag, _ in results:
        status = "PASS" if passed_flag else "FAIL"
        icon = "✅" if passed_flag else "❌"
        print(f"  {icon} {name}: {status}")

    print()
    print(f"총 {total}개 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
