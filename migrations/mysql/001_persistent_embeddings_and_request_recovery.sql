-- Persistent E5 embeddings and upload request recovery for MySQL 8+.
-- This migration is idempotent and safe for existing rows because every new
-- column is nullable. Run it against the configured Aiven database before
-- deploying the application code that reads these columns.

SET @migration_schema := DATABASE();

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'chunks'
          AND column_name = 'embedding_blob'
    ),
    'SELECT 1',
    'ALTER TABLE `chunks` ADD COLUMN `embedding_blob` LONGBLOB NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'chunks'
          AND column_name = 'embedding_dimension'
    ),
    'SELECT 1',
    'ALTER TABLE `chunks` ADD COLUMN `embedding_dimension` INT NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'chunks'
          AND column_name = 'embedding_dtype'
    ),
    'SELECT 1',
    'ALTER TABLE `chunks` ADD COLUMN `embedding_dtype` VARCHAR(32) NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'queries'
          AND column_name = 'request_id'
    ),
    'SELECT 1',
    'ALTER TABLE `queries` ADD COLUMN `request_id` VARCHAR(64) NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'queries'
          AND column_name = 'request_index'
    ),
    'SELECT 1',
    'ALTER TABLE `queries` ADD COLUMN `request_index` INT NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND index_name = 'ix_documents_user_id_source_hash'
    ),
    'SELECT 1',
    'CREATE INDEX `ix_documents_user_id_source_hash` ON `documents` (`user_id`, `source_hash`)'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

-- These columns are new and therefore NULL on every old row. MySQL allows
-- multiple NULL values in a unique index, so existing history cannot collide.
SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = @migration_schema
          AND table_name = 'queries'
          AND index_name = 'uq_queries_user_request_index'
    ),
    'SELECT 1',
    'ALTER TABLE `queries` ADD UNIQUE INDEX `uq_queries_user_request_index` (`user_id`, `request_id`, `request_index`)'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;
