.PHONY: audit audit-lock build check format lint sync sync-faiss test typecheck

ARTIFACT_DIR ?= dist

sync:
	uv sync --frozen --no-install-project
	uv sync --frozen --no-build-isolation

sync-faiss:
	uv sync --frozen --no-install-project --extra faiss
	uv sync --frozen --no-build-isolation --extra faiss

format: sync
	uv run --no-sync ruff format .
	uv run --no-sync ruff check --fix .

lint: sync
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .

typecheck: sync
	uv run --no-sync mypy src

test: sync
	uv run --no-sync pytest

audit-lock: sync
	uv lock --check
	uv run --no-sync python -c 'import pathlib, tomllib; lock = tomllib.loads(pathlib.Path("uv.lock").read_text(encoding="utf-8")); pins = sorted({"{}=={}".format(package["name"], package["version"]) for package in lock["package"] if "registry" in package.get("source", {})}); assert pins; pathlib.Path("/tmp/mira-all-locked-registry.txt").write_text("\n".join(pins) + "\n", encoding="utf-8"); print("exported {} unique registry pins".format(len(pins)))'
	uv run --no-sync pip-audit --strict --no-deps --disable-pip --requirement /tmp/mira-all-locked-registry.txt

audit: sync audit-lock
	uv run --no-sync lint-imports
	uv run --no-sync vulture src/mira
	uv export --frozen --no-dev --no-emit-project --output-file /tmp/mira-runtime-requirements.txt > /dev/null
	uv run --no-sync pip-audit --strict --requirement /tmp/mira-runtime-requirements.txt
	uv export --frozen --all-groups --no-emit-project --output-file /tmp/mira-dev-requirements.txt > /dev/null
	uv run --no-sync pip-audit --strict --requirement /tmp/mira-dev-requirements.txt
	uv export --frozen --extra faiss --no-dev --no-emit-project --output-file /tmp/mira-faiss-requirements.txt > /dev/null
	uv run --no-sync pip-audit --strict --requirement /tmp/mira-faiss-requirements.txt

build: sync
	uv build --no-build-isolation --out-dir "$(ARTIFACT_DIR)"

check: sync
	uv lock --check
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
	uv run --no-sync mypy src
	uv run --no-sync lint-imports
	uv run --no-sync vulture src/mira
	uv run --no-sync pytest
