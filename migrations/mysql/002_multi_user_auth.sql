-- Phase 1 multi-user authentication schema for MySQL 8+.
--
-- This migration is intentionally destructive for the legacy placeholder
-- identity. Existing local-dev-user persistence is disposable and is deleted
-- in foreign-key-safe order after the additive schema changes complete.
--
-- The DDL checks information_schema (or uses CREATE TABLE IF NOT EXISTS), so
-- the migration can be rerun after a complete or interrupted execution.

SET @migration_schema := DATABASE();

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = @migration_schema
          AND table_name = 'users'
          AND column_name = 'password_hash'
    ),
    'SELECT 1',
    'ALTER TABLE `users` ADD COLUMN `password_hash` VARCHAR(255) NULL'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

SET @ddl := IF(
    EXISTS(
        SELECT 1 FROM information_schema.statistics
        WHERE table_schema = @migration_schema
          AND table_name = 'users'
          AND index_name = 'uq_users_email'
    ),
    'SELECT 1',
    'ALTER TABLE `users` ADD UNIQUE INDEX `uq_users_email` (`email`)'
);
PREPARE migration_stmt FROM @ddl;
EXECUTE migration_stmt;
DEALLOCATE PREPARE migration_stmt;

CREATE TABLE IF NOT EXISTS `refresh_sessions` (
    `id` VARCHAR(36) NOT NULL,
    `user_id` VARCHAR(128) NOT NULL,
    `token_hash` BINARY(32) NOT NULL,
    `expires_at` DATETIME(6) NOT NULL,
    `revoked_at` DATETIME(6) NULL,
    `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    `replaced_by_session_id` VARCHAR(36) NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uq_refresh_sessions_token_hash` (`token_hash`),
    KEY `ix_refresh_sessions_expires_at` (`expires_at`),
    KEY `ix_refresh_sessions_user_id_revoked_at_expires_at` (`user_id`, `revoked_at`, `expires_at`),
    CONSTRAINT `fk_refresh_sessions_user_id_users`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_refresh_sessions_replaced_by_session_id`
        FOREIGN KEY (`replaced_by_session_id`) REFERENCES `refresh_sessions` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Delete only data connected to the legacy placeholder identity. The joins
-- defensively remove descendants even if an old row's denormalized user_id is
-- inconsistent with its referenced parent.
SET @legacy_user_id := 'local-dev-user';

START TRANSACTION;

DELETE citation
FROM `citations` AS citation
LEFT JOIN `queries` AS stored_query ON stored_query.id = citation.query_id
LEFT JOIN `documents` AS document ON document.id = citation.document_id
LEFT JOIN `chunks` AS chunk_row ON chunk_row.id = citation.chunk_db_id
WHERE citation.user_id = @legacy_user_id
   OR stored_query.user_id = @legacy_user_id
   OR document.user_id = @legacy_user_id
   OR chunk_row.user_id = @legacy_user_id;

DELETE stored_query
FROM `queries` AS stored_query
LEFT JOIN `documents` AS document ON document.id = stored_query.document_id
WHERE stored_query.user_id = @legacy_user_id
   OR document.user_id = @legacy_user_id;

DELETE chunk_row
FROM `chunks` AS chunk_row
LEFT JOIN `documents` AS document ON document.id = chunk_row.document_id
WHERE chunk_row.user_id = @legacy_user_id
   OR document.user_id = @legacy_user_id;

DELETE FROM `documents`
WHERE user_id = @legacy_user_id;

DELETE FROM `refresh_sessions`
WHERE user_id = @legacy_user_id;

DELETE FROM `users`
WHERE id = @legacy_user_id;

COMMIT;
