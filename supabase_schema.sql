-- =====================================================
-- 재처리 대기열 및 청크 결과 테이블 스키마
-- Supabase SQL Editor에서 실행
-- =====================================================

-- 1. 재처리 대기열 테이블
CREATE TABLE IF NOT EXISTS chunk_retry_queue (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(255) NOT NULL,
    chunk_key VARCHAR(255) NOT NULL,
    product_id INTEGER NOT NULL,
    product_code VARCHAR(255),
    image_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    chunk_width INTEGER,
    chunk_height INTEGER,
    original_chunk_path VARCHAR(500),

    -- 처리 설정
    target_lang VARCHAR(10) DEFAULT 'en',
    prompt TEXT,
    gemini_model VARCHAR(100) DEFAULT 'gemini-2.0-flash-001',
    gemini_api_key TEXT,
    storage_bucket VARCHAR(255) DEFAULT 'translated-images',
    table_name VARCHAR(255),
    output_format VARCHAR(10) DEFAULT 'WEBP',
    output_quality INTEGER DEFAULT 100,
    config_json JSONB,

    -- 상태 관리
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    translated_base64 TEXT,  -- 성공 시 결과 저장

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- 인덱스용 유니크 키
    UNIQUE(batch_id, chunk_key)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_retry_queue_status ON chunk_retry_queue(status);
CREATE INDEX IF NOT EXISTS idx_retry_queue_batch_id ON chunk_retry_queue(batch_id);
CREATE INDEX IF NOT EXISTS idx_retry_queue_product_image ON chunk_retry_queue(product_id, image_index);
CREATE INDEX IF NOT EXISTS idx_retry_queue_created_at ON chunk_retry_queue(created_at);

-- updated_at 자동 업데이트 트리거
CREATE OR REPLACE FUNCTION update_retry_queue_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_retry_queue_updated_at ON chunk_retry_queue;
CREATE TRIGGER trigger_retry_queue_updated_at
    BEFORE UPDATE ON chunk_retry_queue
    FOR EACH ROW
    EXECUTE FUNCTION update_retry_queue_updated_at();


-- 2. 청크 결과 테이블 (병합 트리거용)
CREATE TABLE IF NOT EXISTS chunk_results (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(255) NOT NULL,
    chunk_key VARCHAR(255) NOT NULL,
    product_id INTEGER NOT NULL,
    product_code VARCHAR(255),
    image_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    chunk_width INTEGER,
    chunk_height INTEGER,

    -- 번역 결과
    translated_base64 TEXT,
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('completed', 'failed')),

    -- 타임스탬프
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- 유니크 키 (upsert용)
    UNIQUE(batch_id, chunk_key)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_chunk_results_batch_id ON chunk_results(batch_id);
CREATE INDEX IF NOT EXISTS idx_chunk_results_product_image ON chunk_results(product_id, image_index);
CREATE INDEX IF NOT EXISTS idx_chunk_results_status ON chunk_results(status);

-- updated_at 자동 업데이트 트리거
CREATE OR REPLACE FUNCTION update_chunk_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_chunk_results_updated_at ON chunk_results;
CREATE TRIGGER trigger_chunk_results_updated_at
    BEFORE UPDATE ON chunk_results
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_results_updated_at();


-- 3. 유용한 뷰: 배치별 처리 현황
CREATE OR REPLACE VIEW batch_processing_status AS
SELECT
    batch_id,
    COUNT(*) as total_items,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'processing') as processing,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    MIN(created_at) as first_created,
    MAX(updated_at) as last_updated
FROM chunk_retry_queue
GROUP BY batch_id
ORDER BY first_created DESC;


-- 4. 유용한 뷰: 이미지별 완료 현황
CREATE OR REPLACE VIEW image_completion_status AS
SELECT
    batch_id,
    product_id,
    image_index,
    total_chunks,
    COUNT(*) as completed_chunks,
    CASE
        WHEN COUNT(*) >= total_chunks THEN true
        ELSE false
    END as is_complete
FROM chunk_results
WHERE status = 'completed'
GROUP BY batch_id, product_id, image_index, total_chunks;


-- 5. 오래된 데이터 정리 함수 (선택적)
CREATE OR REPLACE FUNCTION cleanup_old_retry_queue(days_old INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM chunk_retry_queue
    WHERE status IN ('completed', 'failed')
    AND updated_at < NOW() - (days_old || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 사용 예: SELECT cleanup_old_retry_queue(7);  -- 7일 이상된 완료/실패 항목 삭제


-- 6. RLS (Row Level Security) 정책 (필요한 경우)
-- ALTER TABLE chunk_retry_queue ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunk_results ENABLE ROW LEVEL SECURITY;

-- 서비스 역할에 대한 전체 액세스 허용
-- CREATE POLICY "Service role full access" ON chunk_retry_queue
--     FOR ALL USING (true) WITH CHECK (true);
-- CREATE POLICY "Service role full access" ON chunk_results
--     FOR ALL USING (true) WITH CHECK (true);


-- =====================================================
-- Storage 버킷 생성 (Supabase Dashboard에서 수동 생성 필요)
-- =====================================================
-- 1. original-chunks: 원본 청크 이미지 저장 (비공개)
-- 2. translated-images: 번역된 이미지 저장 (공개 가능)
