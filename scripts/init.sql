-- Aegis PostgreSQL Initialization Script
-- This runs automatically on first container start.
-- SQLAlchemy's init_db() also creates tables, so this is a safety net.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Ensure the aegis user has all permissions
GRANT ALL PRIVILEGES ON DATABASE aegis_db TO aegis;
