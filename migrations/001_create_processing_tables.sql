-- =====================================================
-- 이미지 번역 처리 추적 테이블
-- 생성일: 2026-01-31
-- =====================================================

-- 1. image_processing 테이블 (이미지 단위 상태 관리)
CREATE TABLE IF NOT EXISTS image_processing (
  id SERIAL PRIMARY KEY,
  batch_job_id INTEGER REFERENCES batch_jobs(id) ON DELETE CASCADE,
  product_id INTEGER NOT NULL,
  image_index INTEGER NOT NULL,           -- 상품 내 이미지 순서 (0부터)
  original_url TEXT,                      -- 원본 이미지 URL

  status VARCHAR(20) DEFAULT 'pending',
  -- pending: 대기 중
  -- processing: 처리 중
  -- partial: 일부 청크 정상, 나머지 재처리 중
  -- completed: 완료 (병합 완료)
  -- failed: 실패 (3회 재처리 후 원본 대체)

  total_chunks INTEGER DEFAULT 0,         -- 전체 청크 수
  valid_chunks INTEGER DEFAULT 0,         -- 정상 청크 수
  invalid_chunks INTEGER DEFAULT 0,       -- 비정상 청크 수
  replaced_chunks INTEGER DEFAULT 0,      -- 원본으로 대체된 청크 수

  merged_url TEXT,                        -- 최종 병합된 이미지 URL (Supabase Storage)

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(batch_job_id, product_id, image_index)
);

-- 2. chunk_processing 테이블 (청크 단위 상태 관리)
CREATE TABLE IF NOT EXISTS chunk_processing (
  id SERIAL PRIMARY KEY,
  image_processing_id INTEGER REFERENCES image_processing(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,           -- 이미지 내 청크 순서 (0부터)

  status VARCHAR(20) DEFAULT 'pending',
  -- pending: 대기 중
  -- valid: 검증 통과
  -- invalid: 검증 실패 (재처리 필요)
  -- retrying: 재처리 중
  -- replaced: 3회 실패 후 원본으로 대체

  retry_count INTEGER DEFAULT 0,          -- 재처리 횟수 (max 3)

  original_base64 TEXT,                   -- 원본 청크 이미지 (재처리용)
  translated_base64 TEXT,                 -- 번역된 청크 이미지 (정상일 때)

  -- OCR 검증 결과
  validation_result JSONB,
  -- {
  --   "valid": true/false,
  --   "target_lang_ratio": 0.95,
  --   "detected_text": [...],
  --   "reason": "번역 검증 통과: 타겟 언어(en) 비율 95%"
  -- }

  last_error TEXT,                        -- 마지막 에러 메시지

  retry_batch_id INTEGER,                 -- 현재 재처리 중인 배치 ID (batch_jobs.id 참조)

  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(image_processing_id, chunk_index)
);

-- 3. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_image_processing_batch_job_id
  ON image_processing(batch_job_id);

CREATE INDEX IF NOT EXISTS idx_image_processing_status
  ON image_processing(status);

CREATE INDEX IF NOT EXISTS idx_image_processing_product_id
  ON image_processing(product_id);

CREATE INDEX IF NOT EXISTS idx_chunk_processing_image_processing_id
  ON chunk_processing(image_processing_id);

CREATE INDEX IF NOT EXISTS idx_chunk_processing_status
  ON chunk_processing(status);

CREATE INDEX IF NOT EXISTS idx_chunk_processing_retry_batch_id
  ON chunk_processing(retry_batch_id);

-- 재처리 대상 청크 빠르게 조회
CREATE INDEX IF NOT EXISTS idx_chunk_processing_retry_target
  ON chunk_processing(status, retry_count)
  WHERE status = 'invalid' AND retry_count < 3;

-- 4. updated_at 자동 업데이트 트리거
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_image_processing_updated_at ON image_processing;
CREATE TRIGGER update_image_processing_updated_at
  BEFORE UPDATE ON image_processing
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_chunk_processing_updated_at ON chunk_processing;
CREATE TRIGGER update_chunk_processing_updated_at
  BEFORE UPDATE ON chunk_processing
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- 5. RLS (Row Level Security) 비활성화 (서비스 키 사용)
ALTER TABLE image_processing ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_processing ENABLE ROW LEVEL SECURITY;

-- 서비스 역할은 모든 작업 허용
CREATE POLICY "Service role full access on image_processing"
  ON image_processing FOR ALL
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access on chunk_processing"
  ON chunk_processing FOR ALL
  USING (true)
  WITH CHECK (true);

-- 6. 유용한 뷰 생성
CREATE OR REPLACE VIEW v_batch_progress AS
SELECT
  bj.id as batch_job_id,
  bj.status as batch_status,
  bj.table_name,
  COUNT(DISTINCT ip.id) as total_images,
  COUNT(DISTINCT ip.id) FILTER (WHERE ip.status = 'completed') as completed_images,
  COUNT(DISTINCT ip.id) FILTER (WHERE ip.status = 'partial') as partial_images,
  COUNT(DISTINCT ip.id) FILTER (WHERE ip.status = 'failed') as failed_images,
  COUNT(cp.id) as total_chunks,
  COUNT(cp.id) FILTER (WHERE cp.status = 'valid') as valid_chunks,
  COUNT(cp.id) FILTER (WHERE cp.status = 'invalid') as invalid_chunks,
  COUNT(cp.id) FILTER (WHERE cp.status = 'retrying') as retrying_chunks,
  COUNT(cp.id) FILTER (WHERE cp.status = 'replaced') as replaced_chunks
FROM batch_jobs bj
LEFT JOIN image_processing ip ON ip.batch_job_id = bj.id
LEFT JOIN chunk_processing cp ON cp.image_processing_id = ip.id
GROUP BY bj.id, bj.status, bj.table_name;

-- 완료!
COMMENT ON TABLE image_processing IS '이미지 단위 번역 처리 상태 추적';
COMMENT ON TABLE chunk_processing IS '청크 단위 번역 처리 상태 추적';
COMMENT ON VIEW v_batch_progress IS '배치 작업 진행 상황 요약';
