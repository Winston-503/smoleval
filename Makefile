LIB_DIR := crates/smoleval
CLI_DIR := crates/smoleval-cli

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: check
check:  ## Run cargo check (workspace)
	cargo check

.PHONY: build
build:  ## Run cargo build (workspace)
	cargo build

.PHONY: release
release:  ## Build release binary
	cargo build --release -p smoleval-cli

.PHONY: format
format:  ## Run cargo fmt (workspace)
	cargo fmt

.PHONY: lint
lint:  ## Check formatting and run clippy
	cargo fmt --check
	cargo clippy -- -D warnings

.PHONY: dev-lint
dev-lint: format  ## Run format and clippy --fix to auto-fix errors
	cargo clippy --fix --allow-dirty --allow-staged

.PHONY: test
test:  ## Run Rust tests (workspace)
	cargo test

.PHONY: example-smoleval
example-smoleval:  ## Run the smoleval-example binary
	cargo run -p smoleval-example

.PHONY: example-rig-agent
example-rig-agent:  ## Start the Rig agent HTTP server for smoleval-cli
	cargo run -p smoleval-cli-example

.PHONY: example-langchain-agent
example-langchain-agent:  ## Start the LangChain agent HTTP server for smoleval-cli
	uv run crates/smoleval-cli-example/agent.py

.PHONY: example-langgraph-agent-eval
example-langgraph-agent-eval:  ## Run smoleval-cli for LangGraph agent
	cargo run -p smoleval-cli -- \
		--dataset crates/smoleval-cli-example/data/eval_dataset.yaml \
		--agent http://localhost:3826 \
		--concurrency 3 \
		--quiet \
		--format json --output results.json

.PHONY: doc-lib
doc-lib:  ## Run cargo doc for smoleval lib
	cd $(LIB_DIR) && cargo doc --open

.PHONY: doc-cli
doc-cli:  ## Run cargo doc for smoleval-cli
	cd $(CLI_DIR) && cargo doc --open

.PHONY: publish-lib-dry-run
publish-lib-dry-run:  ## Publish smoleval lib in dry run
	cd $(LIB_DIR) && cargo publish --dry-run

.PHONY: publish-lib
publish-lib:  ## Publish smoleval lib to crates.io
	cd $(LIB_DIR) && cargo publish

.PHONY: publish-cli-dry-run
publish-cli-dry-run:  ## Publish smoleval-cli in dry run
	cd $(CLI_DIR) && cargo publish --dry-run

.PHONY: publish-cli
publish-cli:  ## Publish smoleval-cli to crates.io
	cd $(CLI_DIR) && cargo publish

.PHONY: binary-info
binary-info: release  ## Show .rs line counts and release binary size
	@printf "\033[1mRust lines (excluding tests):\033[0m\n"
	@for dir in $(LIB_DIR)/src $(CLI_DIR)/src; do \
		count=$$(find $$dir -name '*.rs' -exec sed '/#\[cfg(test)\]/,$$d' {} \; | grep -cv '^\s*$$'); \
		printf "  %-20s %d lines\n" "$$(basename $$(dirname $$dir)):" "$$count"; \
	done
	@total=$$(find $(LIB_DIR)/src $(CLI_DIR)/src -name '*.rs' -exec sed '/#\[cfg(test)\]/,$$d' {} \; | grep -cv '^\s*$$'); \
		printf "  %-20s %d lines\n" "total:" "$$total"
	@printf "\033[1mRelease binary:\033[0m\n"
	@ls -lh target/release/smoleval | awk '{printf "  size: %s\n", $$5}'
