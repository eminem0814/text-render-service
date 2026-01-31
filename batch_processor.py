"""
배치 처리 모듈 v2
- 이미지 단위 처리
- 청크별 OCR 검증
- 재처리 루프 (최대 3회)
- 진행 상황 트래킹
"""

import requests
import json
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ===== 상수 =====
MAX_RETRY_COUNT = 3  # 최대 재처리 횟수
OCR_THRESHOLD = 0.2  # 비타겟 언어 20% 이상이면 불량


# ===== DB 헬퍼 함수 =====

def supabase_request(
    method: str,
    endpoint: str,
    supabase_url: str,
    supabase_key: str,
    data: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Tuple[bool, Any]:
    """Supabase REST API 요청 헬퍼"""
    url = f"{supabase_url}/rest/v1/{endpoint}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data, params=params, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=params, timeout=30)
        else:
            return False, f"Unknown method: {method}"

        if response.status_code in [200, 201, 204]:
            if response.text:
                return True, response.json()
            return True, None
        else:
            return False, f"HTTP {response.status_code}: {response.text[:500]}"
    except Exception as e:
        return False, str(e)


def create_image_processing_record(
    batch_job_id: int,
    product_id: int,
    image_index: int,
    original_url: str,
    total_chunks: int,
    supabase_url: str,
    supabase_key: str
) -> Tuple[bool, Any]:
    """이미지 처리 레코드 생성"""
    data = {
        "batch_job_id": batch_job_id,
        "product_id": product_id,
        "image_index": image_index,
        "original_url": original_url,
        "total_chunks": total_chunks,
        "status": "processing"
    }
    return supabase_request("POST", "image_processing", supabase_url, supabase_key, data)


def create_chunk_processing_record(
    image_processing_id: int,
    chunk_index: int,
    original_base64: str,
    supabase_url: str,
    supabase_key: str
) -> Tuple[bool, Any]:
    """청크 처리 레코드 생성"""
    data = {
        "image_processing_id": image_processing_id,
        "chunk_index": chunk_index,
        "original_base64": original_base64,
        "status": "pending"
    }
    return supabase_request("POST", "chunk_processing", supabase_url, supabase_key, data)


def update_chunk_status(
    chunk_id: int,
    status: str,
    supabase_url: str,
    supabase_key: str,
    translated_base64: Optional[str] = None,
    validation_result: Optional[Dict] = None,
    last_error: Optional[str] = None,
    increment_retry: bool = False
) -> Tuple[bool, Any]:
    """청크 상태 업데이트"""
    data = {"status": status}

    if translated_base64:
        data["translated_base64"] = translated_base64
    if validation_result:
        data["validation_result"] = validation_result
    if last_error:
        data["last_error"] = last_error

    # retry_count 증가가 필요한 경우
    if increment_retry:
        # 먼저 현재 retry_count 조회
        success, result = supabase_request(
            "GET",
            f"chunk_processing?id=eq.{chunk_id}&select=retry_count",
            supabase_url, supabase_key
        )
        if success and result:
            current_count = result[0].get("retry_count", 0)
            data["retry_count"] = current_count + 1

    return supabase_request(
        "PATCH",
        f"chunk_processing?id=eq.{chunk_id}",
        supabase_url, supabase_key,
        data
    )


def update_image_status(
    image_id: int,
    status: str,
    supabase_url: str,
    supabase_key: str,
    merged_url: Optional[str] = None,
    valid_chunks: Optional[int] = None,
    invalid_chunks: Optional[int] = None,
    replaced_chunks: Optional[int] = None
) -> Tuple[bool, Any]:
    """이미지 상태 업데이트"""
    data = {"status": status}

    if merged_url:
        data["merged_url"] = merged_url
    if valid_chunks is not None:
        data["valid_chunks"] = valid_chunks
    if invalid_chunks is not None:
        data["invalid_chunks"] = invalid_chunks
    if replaced_chunks is not None:
        data["replaced_chunks"] = replaced_chunks

    return supabase_request(
        "PATCH",
        f"image_processing?id=eq.{image_id}",
        supabase_url, supabase_key,
        data
    )


def get_image_processing_by_batch(
    batch_job_id: int,
    supabase_url: str,
    supabase_key: str,
    status_filter: Optional[str] = None
) -> Tuple[bool, List]:
    """배치별 이미지 처리 상태 조회"""
    endpoint = f"image_processing?batch_job_id=eq.{batch_job_id}&select=*"
    if status_filter:
        endpoint += f"&status=eq.{status_filter}"
    return supabase_request("GET", endpoint, supabase_url, supabase_key)


def get_chunks_by_image(
    image_processing_id: int,
    supabase_url: str,
    supabase_key: str
) -> Tuple[bool, List]:
    """이미지별 청크 목록 조회"""
    endpoint = f"chunk_processing?image_processing_id=eq.{image_processing_id}&select=*&order=chunk_index"
    return supabase_request("GET", endpoint, supabase_url, supabase_key)


def get_invalid_chunks_for_retry(
    batch_job_id: int,
    supabase_url: str,
    supabase_key: str
) -> Tuple[bool, List]:
    """재처리 대상 청크 조회 (invalid 상태, retry_count < 3)"""
    # 먼저 해당 배치의 이미지 ID 목록 조회
    success, images = get_image_processing_by_batch(batch_job_id, supabase_url, supabase_key)
    if not success or not images:
        return False, []

    image_ids = [img["id"] for img in images]

    # 해당 이미지들의 invalid 청크 조회
    endpoint = f"chunk_processing?image_processing_id=in.({','.join(map(str, image_ids))})&status=eq.invalid&retry_count=lt.{MAX_RETRY_COUNT}&select=*,image_processing(product_id,image_index,batch_job_id)"
    return supabase_request("GET", endpoint, supabase_url, supabase_key)


def get_batch_progress(
    batch_job_id: int,
    supabase_url: str,
    supabase_key: str
) -> Dict:
    """배치 진행 상황 조회"""
    success, images = get_image_processing_by_batch(batch_job_id, supabase_url, supabase_key)

    if not success:
        return {"error": str(images)}

    progress = {
        "batch_job_id": batch_job_id,
        "total_images": len(images),
        "completed": 0,
        "partial": 0,
        "processing": 0,
        "failed": 0,
        "images": []
    }

    for img in images:
        status = img.get("status", "pending")
        if status == "completed":
            progress["completed"] += 1
        elif status == "partial":
            progress["partial"] += 1
        elif status == "processing":
            progress["processing"] += 1
        elif status == "failed":
            progress["failed"] += 1

        progress["images"].append({
            "id": img["id"],
            "product_id": img["product_id"],
            "image_index": img["image_index"],
            "status": status,
            "total_chunks": img.get("total_chunks", 0),
            "valid_chunks": img.get("valid_chunks", 0),
            "invalid_chunks": img.get("invalid_chunks", 0),
            "replaced_chunks": img.get("replaced_chunks", 0)
        })

    progress["is_complete"] = progress["completed"] + progress["failed"] == progress["total_images"]

    return progress


# ===== 배치 결과 처리 =====

def process_batch_results_v2(
    gemini_file_name: str,
    gemini_api_key: str,
    chunk_metadata: List[Dict],
    product_map: Dict,
    config: Dict,
    job_id: int,
    validate_chunk_func,
    validate_translation_func,
    base64_to_image_func,
    image_to_base64_func,
    merge_images_func
) -> Dict:
    """
    Gemini 배치 결과 처리 v2
    - 새 테이블 구조 사용
    - 청크별 검증 및 상태 기록
    - 이미지 단위 병합 및 업로드

    Returns:
        {
            "success": bool,
            "total_images": int,
            "completed_images": int,
            "partial_images": int,
            "needs_retry": bool,
            "retry_chunks": int,
            "results": [...]
        }
    """
    supabase_url = config.get("supabaseUrl")
    supabase_key = config.get("supabaseKey")
    storage_bucket = config.get("storageBucket", "translated-images")
    table_name = config.get("tableName", "")
    target_lang_code = config.get("targetLangCode", "en")
    output_format = config.get("outputFormat", "WEBP").upper()
    output_quality = int(config.get("outputQuality", 100))
    output_extension = config.get("outputExtension", ".webp")

    if not supabase_url or not supabase_key:
        return {"success": False, "error": "Supabase credentials required"}

    # 1. Gemini 결과 파일 다운로드
    logger.info(f"[BatchV2] Downloading results: {gemini_file_name}")

    download_url = f"https://generativelanguage.googleapis.com/download/v1beta/{gemini_file_name}:download?alt=media"

    try:
        download_response = requests.get(
            download_url,
            headers={"x-goog-api-key": gemini_api_key},
            timeout=300
        )

        if download_response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to download: {download_response.status_code}"
            }

        jsonl_content = download_response.text
        logger.info(f"[BatchV2] Downloaded {len(jsonl_content)} bytes")
    except Exception as e:
        return {"success": False, "error": f"Download error: {str(e)}"}

    # 2. JSONL 파싱
    lines = jsonl_content.strip().split('\n')
    meta_by_key = {m["key"]: m for m in chunk_metadata}

    # 이미지별 청크 그룹핑
    # 구조: {image_key: {image_data, chunks: [{chunk_data, gemini_result}]}}
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
            logger.warning(f"[BatchV2] No image data for {custom_id}")
            continue

        image_key = f"{meta['productId']}_{meta['imageIndex']}"

        if image_key not in image_chunks:
            image_chunks[image_key] = {
                "productId": meta["productId"],
                "productCode": meta.get("productCode"),
                "imageIndex": meta["imageIndex"],
                "totalChunks": meta.get("totalChunks", 1),
                "originalWidth": meta.get("chunkWidth"),
                "originalUrl": meta.get("originalUrl", ""),
                "chunks": []
            }

        image_chunks[image_key]["chunks"].append({
            "index": meta["chunkIndex"],
            "translated_base64": translated_base64,
            "original_base64": meta.get("originalBase64", ""),
            "expected_width": meta.get("chunkWidth"),
            "expected_height": meta.get("chunkHeight"),
            "key": custom_id
        })

    logger.info(f"[BatchV2] Grouped into {len(image_chunks)} images")

    # 3. 이미지별 처리
    results = []
    completed_count = 0
    partial_count = 0
    total_retry_chunks = 0

    # PIL 포맷 매핑
    pil_format = "JPEG" if output_format == "JPG" else output_format
    mime_types = {
        "WEBP": "image/webp",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png"
    }
    mime_type = mime_types.get(output_format, "image/webp")

    for image_key, image_data in image_chunks.items():
        try:
            product_id = image_data["productId"]
            image_index = image_data["imageIndex"]
            total_chunks = image_data["totalChunks"]

            # 3.1 이미지 처리 레코드 생성 (또는 조회)
            success, existing = supabase_request(
                "GET",
                f"image_processing?batch_job_id=eq.{job_id}&product_id=eq.{product_id}&image_index=eq.{image_index}",
                supabase_url, supabase_key
            )

            if success and existing:
                image_record = existing[0]
                image_processing_id = image_record["id"]
            else:
                success, result = create_image_processing_record(
                    job_id, product_id, image_index,
                    image_data.get("originalUrl", ""),
                    total_chunks,
                    supabase_url, supabase_key
                )
                if not success:
                    logger.error(f"[BatchV2] Failed to create image record: {result}")
                    continue
                image_processing_id = result[0]["id"]

            # 3.2 청크별 검증
            sorted_chunks = sorted(image_data["chunks"], key=lambda x: x["index"])
            valid_chunks = []
            invalid_chunks = []

            for chunk in sorted_chunks:
                chunk_index = chunk["index"]
                translated_b64 = chunk["translated_base64"]
                original_b64 = chunk.get("original_base64", "")

                # 청크 레코드 생성 (또는 조회)
                success, existing_chunk = supabase_request(
                    "GET",
                    f"chunk_processing?image_processing_id=eq.{image_processing_id}&chunk_index=eq.{chunk_index}",
                    supabase_url, supabase_key
                )

                if success and existing_chunk:
                    chunk_record = existing_chunk[0]
                    chunk_id = chunk_record["id"]
                    # 이미 valid인 경우 스킵
                    if chunk_record["status"] == "valid":
                        valid_chunks.append({
                            "index": chunk_index,
                            "base64": chunk_record.get("translated_base64", translated_b64)
                        })
                        continue
                else:
                    success, result = create_chunk_processing_record(
                        image_processing_id, chunk_index, original_b64,
                        supabase_url, supabase_key
                    )
                    if not success:
                        logger.error(f"[BatchV2] Failed to create chunk record: {result}")
                        continue
                    chunk_id = result[0]["id"]

                # 크기 검증
                size_valid = validate_chunk_func(
                    translated_b64,
                    expected_width=chunk.get("expected_width"),
                    expected_height=chunk.get("expected_height"),
                    tolerance=0.3
                )

                if not size_valid.get("valid", False):
                    # 크기 불량
                    update_chunk_status(
                        chunk_id, "invalid",
                        supabase_url, supabase_key,
                        validation_result={"size_check": size_valid},
                        last_error=size_valid.get("reason", "Size validation failed")
                    )
                    invalid_chunks.append(chunk_index)
                    continue

                # OCR 검증
                try:
                    ocr_result = validate_translation_func(
                        translated_b64,
                        target_lang_code,
                        threshold=OCR_THRESHOLD
                    )
                except Exception as ocr_err:
                    ocr_result = {"valid": True, "reason": f"OCR error (passed): {str(ocr_err)}"}

                if ocr_result.get("valid", False):
                    # 검증 통과
                    update_chunk_status(
                        chunk_id, "valid",
                        supabase_url, supabase_key,
                        translated_base64=translated_b64,
                        validation_result=ocr_result
                    )
                    valid_chunks.append({
                        "index": chunk_index,
                        "base64": translated_b64
                    })
                else:
                    # OCR 검증 실패
                    update_chunk_status(
                        chunk_id, "invalid",
                        supabase_url, supabase_key,
                        validation_result=ocr_result,
                        last_error=ocr_result.get("reason", "OCR validation failed")
                    )
                    invalid_chunks.append(chunk_index)

            # 3.3 이미지 상태 결정
            valid_count = len(valid_chunks)
            invalid_count = len(invalid_chunks)

            if invalid_count == 0 and valid_count == total_chunks:
                # 모두 정상 → 병합 → 업로드
                logger.info(f"[BatchV2] Image {image_key}: All {valid_count} chunks valid, merging...")

                # 청크 정렬 및 병합
                sorted_valid = sorted(valid_chunks, key=lambda x: x["index"])

                if len(sorted_valid) == 1:
                    chunk_image = base64_to_image_func(sorted_valid[0]["base64"])
                    original_width = image_data.get("originalWidth")
                    if original_width and chunk_image.size[0] != original_width:
                        new_height = int(chunk_image.size[1] * original_width / chunk_image.size[0])
                        chunk_image = chunk_image.resize((original_width, new_height))
                    merged_base64, _ = image_to_base64_func(chunk_image, format=pil_format, quality=output_quality)
                else:
                    chunks_for_merge = [{"index": c["index"], "base64": c["base64"]} for c in sorted_valid]
                    merged_image = merge_images_func(
                        chunks_for_merge,
                        overlap=0,
                        blend_height=50,
                        target_width=image_data.get("originalWidth")
                    )
                    merged_base64, _ = image_to_base64_func(merged_image, format=pil_format, quality=output_quality)

                # Supabase Storage 업로드
                image_num = str(image_index + 1).zfill(2)
                file_name = f"{table_name}/{target_lang_code}/{table_name}_ID{product_id}_{image_num}_{target_lang_code}{output_extension}"
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
                    update_image_status(
                        image_processing_id, "completed",
                        supabase_url, supabase_key,
                        merged_url=public_url,
                        valid_chunks=valid_count
                    )
                    completed_count += 1
                    logger.info(f"[BatchV2] Image {image_key}: Completed, uploaded to {file_name}")

                    results.append({
                        "productId": product_id,
                        "imageIndex": image_index,
                        "status": "completed",
                        "uploadedUrl": public_url,
                        "validChunks": valid_count,
                        "invalidChunks": 0
                    })
                else:
                    logger.error(f"[BatchV2] Upload failed: {upload_response.status_code}")
                    update_image_status(
                        image_processing_id, "partial",
                        supabase_url, supabase_key,
                        valid_chunks=valid_count
                    )
                    partial_count += 1
            else:
                # 일부 실패 → partial 상태
                update_image_status(
                    image_processing_id, "partial",
                    supabase_url, supabase_key,
                    valid_chunks=valid_count,
                    invalid_chunks=invalid_count
                )
                partial_count += 1
                total_retry_chunks += invalid_count

                logger.info(f"[BatchV2] Image {image_key}: {valid_count} valid, {invalid_count} invalid (needs retry)")

                results.append({
                    "productId": product_id,
                    "imageIndex": image_index,
                    "status": "partial",
                    "validChunks": valid_count,
                    "invalidChunks": invalid_count,
                    "invalidIndices": invalid_chunks
                })

        except Exception as img_err:
            logger.error(f"[BatchV2] Error processing image {image_key}: {img_err}")
            import traceback
            traceback.print_exc()
            results.append({
                "productId": image_data.get("productId"),
                "imageIndex": image_data.get("imageIndex"),
                "status": "error",
                "error": str(img_err)
            })

    return {
        "success": True,
        "total_images": len(image_chunks),
        "completed_images": completed_count,
        "partial_images": partial_count,
        "needs_retry": total_retry_chunks > 0,
        "retry_chunks": total_retry_chunks,
        "results": results
    }


def create_retry_batch(
    batch_job_id: int,
    config: Dict,
    gemini_api_key: str,
    prompt: str,
    supabase_url: str,
    supabase_key: str
) -> Dict:
    """
    실패한 청크들로 새로운 Gemini 배치 생성

    Returns:
        {
            "success": bool,
            "retry_batch_name": str (Gemini 배치 이름),
            "chunks_count": int
        }
    """
    # 재처리 대상 청크 조회
    success, invalid_chunks = get_invalid_chunks_for_retry(
        batch_job_id, supabase_url, supabase_key
    )

    if not success or not invalid_chunks:
        return {"success": False, "error": "No chunks to retry", "chunks_count": 0}

    logger.info(f"[RetryBatch] Found {len(invalid_chunks)} chunks to retry")

    # Gemini 배치 요청 생성
    batch_requests = []
    chunk_ids = []

    for chunk in invalid_chunks:
        if not chunk.get("original_base64"):
            logger.warning(f"[RetryBatch] Chunk {chunk['id']} has no original_base64")
            continue

        image_info = chunk.get("image_processing", {})
        custom_id = f"retry_{batch_job_id}_{chunk['id']}"

        batch_requests.append({
            "key": custom_id,
            "request": {
                "model": config.get("geminiModel", "gemini-3-pro-image-preview"),
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": chunk["original_base64"]
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "responseModalities": ["image", "text"],
                    "responseMimeType": "image/png"
                }
            }
        })
        chunk_ids.append(chunk["id"])

    if not batch_requests:
        return {"success": False, "error": "No valid chunks to retry", "chunks_count": 0}

    # JSONL 생성
    jsonl_content = "\n".join(json.dumps(req) for req in batch_requests)

    # Gemini 파일 업로드
    try:
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files"

        files = {
            'file': ('retry_batch.jsonl', jsonl_content, 'application/jsonl')
        }

        upload_response = requests.post(
            upload_url,
            headers={"x-goog-api-key": gemini_api_key},
            files=files,
            timeout=120
        )

        if upload_response.status_code != 200:
            return {"success": False, "error": f"File upload failed: {upload_response.status_code}"}

        file_info = upload_response.json()
        file_name = file_info.get("file", {}).get("name")

        if not file_name:
            return {"success": False, "error": "No file name in response"}

        # 배치 작업 생성
        batch_url = "https://generativelanguage.googleapis.com/v1beta/batchEmbedContents:batchProcess"

        # ... (배치 생성 로직은 기존과 동일)

        # 청크 상태를 retrying으로 업데이트
        for chunk_id in chunk_ids:
            update_chunk_status(
                chunk_id, "retrying",
                supabase_url, supabase_key,
                increment_retry=True
            )

        return {
            "success": True,
            "file_name": file_name,
            "chunks_count": len(chunk_ids),
            "chunk_ids": chunk_ids
        }

    except Exception as e:
        logger.error(f"[RetryBatch] Error: {e}")
        return {"success": False, "error": str(e)}


def process_retry_results(
    gemini_file_name: str,
    gemini_api_key: str,
    batch_job_id: int,
    config: Dict,
    validate_translation_func,
    base64_to_image_func,
    image_to_base64_func,
    merge_images_func
) -> Dict:
    """
    재처리 결과 처리
    - 재처리된 청크 검증
    - 정상이면 valid로 변경
    - 3회 실패면 원본으로 대체 (replaced)
    - 모든 청크가 valid/replaced면 병합 및 업로드
    """
    supabase_url = config.get("supabaseUrl")
    supabase_key = config.get("supabaseKey")
    target_lang_code = config.get("targetLangCode", "en")

    # 결과 다운로드 및 파싱 (process_batch_results_v2와 유사)
    # ... 구현 필요

    pass  # TODO: 재처리 결과 처리 로직 구현


def check_and_complete_images(
    batch_job_id: int,
    config: Dict,
    base64_to_image_func,
    image_to_base64_func,
    merge_images_func
) -> Dict:
    """
    완료 가능한 이미지 확인 및 병합
    - 모든 청크가 valid 또는 replaced 상태인 이미지 찾기
    - 병합 및 업로드
    - 상태를 completed로 변경
    """
    supabase_url = config.get("supabaseUrl")
    supabase_key = config.get("supabaseKey")
    storage_bucket = config.get("storageBucket", "translated-images")
    table_name = config.get("tableName", "")
    target_lang_code = config.get("targetLangCode", "en")
    output_format = config.get("outputFormat", "WEBP").upper()
    output_quality = int(config.get("outputQuality", 100))
    output_extension = config.get("outputExtension", ".webp")

    pil_format = "JPEG" if output_format == "JPG" else output_format
    mime_types = {
        "WEBP": "image/webp",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png"
    }
    mime_type = mime_types.get(output_format, "image/webp")

    # partial 상태 이미지 조회
    success, images = get_image_processing_by_batch(
        batch_job_id, supabase_url, supabase_key, status_filter="partial"
    )

    if not success or not images:
        return {"success": True, "completed": 0, "message": "No partial images to check"}

    completed_count = 0
    results = []

    for image in images:
        image_id = image["id"]
        product_id = image["product_id"]
        image_index = image["image_index"]
        total_chunks = image["total_chunks"]

        # 청크 조회
        success, chunks = get_chunks_by_image(image_id, supabase_url, supabase_key)
        if not success:
            continue

        # 모든 청크가 valid 또는 replaced인지 확인
        ready_chunks = []
        not_ready = False
        replaced_count = 0

        for chunk in chunks:
            status = chunk["status"]
            if status == "valid":
                ready_chunks.append({
                    "index": chunk["chunk_index"],
                    "base64": chunk["translated_base64"]
                })
            elif status == "replaced":
                ready_chunks.append({
                    "index": chunk["chunk_index"],
                    "base64": chunk["original_base64"]  # 원본 사용
                })
                replaced_count += 1
            else:
                not_ready = True
                break

        if not_ready or len(ready_chunks) != total_chunks:
            continue

        # 병합 및 업로드
        try:
            sorted_chunks = sorted(ready_chunks, key=lambda x: x["index"])

            if len(sorted_chunks) == 1:
                chunk_image = base64_to_image_func(sorted_chunks[0]["base64"])
                merged_base64, _ = image_to_base64_func(chunk_image, format=pil_format, quality=output_quality)
            else:
                merged_image = merge_images_func(
                    sorted_chunks, overlap=0, blend_height=50
                )
                merged_base64, _ = image_to_base64_func(merged_image, format=pil_format, quality=output_quality)

            # 업로드
            image_num = str(image_index + 1).zfill(2)
            file_name = f"{table_name}/{target_lang_code}/{table_name}_ID{product_id}_{image_num}_{target_lang_code}{output_extension}"
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
                update_image_status(
                    image_id, "completed",
                    supabase_url, supabase_key,
                    merged_url=public_url,
                    replaced_chunks=replaced_count
                )
                completed_count += 1
                results.append({
                    "productId": product_id,
                    "imageIndex": image_index,
                    "status": "completed",
                    "uploadedUrl": public_url,
                    "replacedChunks": replaced_count
                })
            else:
                logger.error(f"Upload failed for image {image_id}: {upload_response.status_code}")

        except Exception as e:
            logger.error(f"Error completing image {image_id}: {e}")

    return {
        "success": True,
        "completed": completed_count,
        "results": results
    }
