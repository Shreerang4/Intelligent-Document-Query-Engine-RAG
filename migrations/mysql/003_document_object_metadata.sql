-- Private source-PDF object metadata for MySQL 8+.
-- This phase is additive: current ingestion routes and document status values
-- are intentionally unchanged. Existing rows receive NULL metadata.

SET @migration_schema := DATABASE();

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND column_name = 'object_key'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ADD COLUMN `object_key` VARCHAR(512) NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND column_name = 'content_type'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ADD COLUMN `content_type` VARCHAR(255) NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND column_name = 'byte_size'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ADD COLUMN `byte_size` BIGINT NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND index_name = 'uq_documents_object_key'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ADD UNIQUE INDEX `uq_documents_object_key` (`object_key`)'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;
