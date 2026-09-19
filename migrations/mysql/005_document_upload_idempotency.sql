-- Owned idempotency key for asynchronous document uploads on MySQL 8+.
-- Existing rows remain NULL; MySQL permits multiple NULLs in a unique index.

SET @migration_schema := DATABASE();

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND column_name = 'upload_request_id'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ADD COLUMN `upload_request_id` VARCHAR(36) NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = @migration_schema
          AND table_name = 'documents'
          AND index_name = 'uq_documents_user_upload_request_id'
    ),
    'SELECT 1',
    'ALTER TABLE `documents` ADD UNIQUE INDEX `uq_documents_user_upload_request_id` (`user_id`, `upload_request_id`)'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;
