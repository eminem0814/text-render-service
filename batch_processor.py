"""
배치 처리 모듈 v3
- 스트리밍 방식으로 메모리 최적화
- 이미지 단위 처리
- 청크별 OCR 검증 (경량 모델)
- 재처리 루프 (최대 3회)
- 진행 상황 트래킹
"""

import requests
import json
import logging
import base64
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ===== 상수 =====
MAX_RETRY_COUNT = 3  # 최대 재처리 횟수
ORIGINAL_CHUNKS_BUCKET = "original-chunks"  # 원본 청크 저장 버킷


def fetch_original_chunk_from_storage(
    chunk_key: str,
    batch_id: str,
    supabase_url: str,
    supabase_key: str
) -> str:
    """Supabase Storage에서 원본 청크 base64 조회"""
    try:
        storage_path = f"{batch_id}/{chunk_key}.jpg"
        download_url = f"{supabase_url}/storage/v1/object/{ORIGINAL_CHUNKS_BUCKET}/{storage_path}"

        response = requests.get(
            download_url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            },
            timeout=30
        )

        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            logger.warning(f"Failed to fetch original chunk {chunk_key}: {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error fetching original chunk {chunk_key}: {e}")
        return ""


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
    """재처리 대상 청크 조회 (invalid/pending/retrying 상태)"""
    # 먼저 해당 배치의 이미지 ID 목록 조회
    success, images = get_image_processing_by_batch(batch_job_id, supabase_url, supabase_key)
    if not success or not images:
        return False, []

    image_ids = [img["id"] for img in images]

    # 해당 이미지들의 불량 청크 조회 (retrying 포함, retry_count 필터 없음)
    # create-retry-batch에서 retry_count >= 3이면 replaced 처리
    endpoint = f"chunk_processing?image_processing_id=in.({','.join(map(str, image_ids))})&status=in.(invalid,pending,retrying)&select=*,image_processing(product_id,image_index,batch_job_id)"
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
