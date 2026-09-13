-- Normalize document ingestion lifecycle states for MySQL 8+.
-- Existing completed documents used the legacy `ingested` value.

UPDATE `documents`
SET `status` = 'ready'
WHERE `status` = 'ingested';

SET @migration_schema := DATABASE();

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND column_name = 'status'
          AND column_default = 'queued'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ALTER COLUMN `status` SET DEFAULT ''queued'''
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;
