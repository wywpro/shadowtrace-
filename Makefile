# Prefer the project venv when present.  Keep this resolver before every target
# so all Python tooling is run from the same interpreter via ``python -m``.
PYTHON ?= $(shell if [ -x "$(CURDIR)/backend/.venv/bin/python" ]; then echo "$(CURDIR)/backend/.venv/bin/python"; else echo python3; fi)

WORKTREE_ID ?= $(shell printf '%s' "$(CURDIR)" | cksum | cut -d ' ' -f 1)
COMPOSE_PROJECT_NAME ?= shadowtrace-$(WORKTREE_ID)
POSTGRES_PORT ?= 5432
REDIS_PORT ?= 6379
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173

COMPOSE_FILE := $(CURDIR)/infra/docker-compose.yml
COMPOSE := COMPOSE_PROJECT_NAME="$(COMPOSE_PROJECT_NAME)" \
	POSTGRES_PORT="$(POSTGRES_PORT)" REDIS_PORT="$(REDIS_PORT)" \
	BACKEND_PORT="$(BACKEND_PORT)" FRONTEND_PORT="$(FRONTEND_PORT)" \
	docker compose --project-name "$(COMPOSE_PROJECT_NAME)" \
	-f "$(COMPOSE_FILE)"

INTEGRATION_PROJECT_NAME ?= $(COMPOSE_PROJECT_NAME)-integration
CI_TEST_PROJECT_NAME ?= $(COMPOSE_PROJECT_NAME)-ci-test
CI_BUILD_PROJECT_PREFIX ?= $(COMPOSE_PROJECT_NAME)-ci-build

# Host-side URLs for tests that talk to Compose postgres/redis from the workstation / CI runner.
CI_DATABASE_URL ?= postgresql+asyncpg://shadowtrace:shadowtrace@localhost:$(POSTGRES_PORT)/shadowtrace
CI_REDIS_URL ?= redis://localhost:$(REDIS_PORT)/0

.PHONY: up down test lint fmt migrate migrate-down load-kb integration-test test-tools ci-lint ci-test ci-build

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

# Apply / roll back the database schema. Override DATABASE_URL to target a host
# (e.g. DATABASE_URL=postgresql+asyncpg://shadowtrace:shadowtrace@localhost:5432/shadowtrace).
migrate:
	cd backend && $(PYTHON) -m alembic upgrade head

migrate-down:
	cd backend && $(PYTHON) -m alembic downgrade base

load-kb:
	cd backend && $(PYTHON) -m scripts.load_attack_kb
	cd backend && $(PYTHON) -m scripts.load_case_kb

test:
	cd backend && $(PYTHON) -m pytest tests/test_infra/test_health.py -v

lint:
	cd backend && $(PYTHON) -m ruff check app tests && $(PYTHON) -m mypy app

fmt:
	cd backend && $(PYTHON) -m ruff check --fix app tests && $(PYTHON) -m ruff format app tests

# --- ISSUE-025 tool-system integration quality gate ---------------------- #
# In-memory Registry/Executor/Mock chains + unit tool tests.
# - Excludes `@pytest.mark.integration` (needs Dockerized Postgres/Redis).
# - Enforces statement coverage >= 80% on app.tools + app.providers.tools.
# - Expected runtime: well under 3 minutes (typically ~30s locally).
# Equivalent:
#   cd backend && pytest tests/test_tools/ tests/integration/test_tool_system.py \
#     -v -m "not integration" --cov=app.tools --cov=app.providers.tools \
#     --cov-fail-under=80
test-tools:
	cd backend && $(PYTHON) -m pytest tests/test_tools/ \
		tests/integration/test_tool_system.py -v -m "not integration" \
		--cov=app.tools --cov=app.providers.tools \
		--cov-report=term-missing --cov-fail-under=80

# --- ISSUE-017 data-foundation integration quality gate ------------------ #
integration-test:
	@set -eu; \
	project="$(INTEGRATION_PROJECT_NAME)"; \
	compose() { \
		COMPOSE_PROJECT_NAME="$$project" \
		POSTGRES_PORT="$(POSTGRES_PORT)" REDIS_PORT="$(REDIS_PORT)" \
		BACKEND_PORT="$(BACKEND_PORT)" FRONTEND_PORT="$(FRONTEND_PORT)" \
		docker compose --project-name "$$project" \
			-f "$(COMPOSE_FILE)" "$$@"; \
	}; \
	cleanup() { \
		status=$$?; \
		trap - EXIT INT TERM; \
		if [ "$$status" -ne 0 ]; then \
			compose ps -a || true; \
			compose logs --no-color postgres redis || true; \
		fi; \
		compose down --volumes --remove-orphans || true; \
		exit "$$status"; \
	}; \
	trap cleanup EXIT INT TERM; \
	compose up -d --wait --wait-timeout 120 postgres redis; \
	cd "$(CURDIR)/backend"; \
	DATABASE_URL="$(CI_DATABASE_URL)" REDIS_URL="$(CI_REDIS_URL)" \
		$(PYTHON) -m pytest tests/integration -m integration -v

# --- ISSUE-009 local / CI parity gates ------------------------------------ #
ci-lint:
	cd backend && $(PYTHON) -m pip install -e ".[dev]" -q
	cd backend && $(PYTHON) -m ruff check app tests
	cd backend && $(PYTHON) -m ruff format --check app tests
	cd backend && $(PYTHON) -m mypy app
	cd frontend && (corepack enable && corepack prepare pnpm@9.15.9 --activate || true)
	cd frontend && pnpm install --frozen-lockfile
	cd frontend && pnpm lint
	cd frontend && pnpm typecheck

ci-test:
	cd backend && $(PYTHON) -m pip install -e ".[dev]" -q
	@set -eu; \
	project="$(CI_TEST_PROJECT_NAME)"; \
	compose() { \
		COMPOSE_PROJECT_NAME="$$project" \
		POSTGRES_PORT="$(POSTGRES_PORT)" REDIS_PORT="$(REDIS_PORT)" \
		BACKEND_PORT="$(BACKEND_PORT)" FRONTEND_PORT="$(FRONTEND_PORT)" \
		docker compose --project-name "$$project" \
			-f "$(COMPOSE_FILE)" "$$@"; \
	}; \
	cleanup() { \
		status=$$?; \
		trap - EXIT INT TERM; \
		if [ "$$status" -ne 0 ]; then \
			compose ps -a || true; \
			compose logs --no-color postgres redis || true; \
		fi; \
		compose down --volumes --remove-orphans || true; \
		exit "$$status"; \
	}; \
	trap cleanup EXIT INT TERM; \
	compose up -d --wait --wait-timeout 120 postgres redis; \
	cd "$(CURDIR)/backend"; \
	DATABASE_URL="$(CI_DATABASE_URL)" REDIS_URL="$(CI_REDIS_URL)" \
		$(PYTHON) -m pytest --cov=app --cov-report=term --cov-report=xml:coverage.xml

ci-build:
	cd frontend && (corepack enable && corepack prepare pnpm@9.15.9 --activate || true)
	cd frontend && pnpm install --frozen-lockfile
	cd frontend && pnpm build
	@set -e; \
	project="$(CI_BUILD_PROJECT_PREFIX)-$$(date +%s)-$$$$"; \
	compose() { \
		COMPOSE_PROJECT_NAME="$$project" \
		POSTGRES_PORT="$(POSTGRES_PORT)" REDIS_PORT="$(REDIS_PORT)" \
		BACKEND_PORT="$(BACKEND_PORT)" FRONTEND_PORT="$(FRONTEND_PORT)" \
		docker compose --project-name "$$project" \
			-f "$(COMPOSE_FILE)" "$$@"; \
	}; \
	cleanup() { \
		status=$$?; \
		trap - EXIT INT TERM; \
		if [ "$$status" -ne 0 ]; then \
			compose ps -a || true; \
			compose logs --no-color postgres redis backend frontend || true; \
		fi; \
		compose down --volumes --remove-orphans || true; \
		exit "$$status"; \
	}; \
	trap cleanup EXIT INT TERM; \
	compose build; \
	compose up -d --wait --wait-timeout 180; \
	for service in postgres redis backend frontend; do \
		container_id=$$(compose ps -q "$$service"); \
		if [ -z "$$container_id" ]; then \
			echo "$$service container is missing"; \
			exit 1; \
		fi; \
		health=$$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "$$container_id"); \
		if [ "$$health" != "healthy" ]; then \
			echo "$$service is not healthy: $$health"; \
			exit 1; \
		fi; \
	done; \
	compose ps; \
	curl --fail --show-error --silent \
		"http://127.0.0.1:$(BACKEND_PORT)/api/v1/health" >/dev/null; \
	curl --fail --show-error --silent \
		"http://127.0.0.1:$(FRONTEND_PORT)/health" >/dev/null
