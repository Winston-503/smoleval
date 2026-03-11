LIB_DIR := crates/smoleval
CLI_DIR := crates/smoleval-cli

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: check
check:  ## Run cargo check (workspace)
	cargo check

.PHONY: build
build:  ## Run cargo build (workspace)
	cargo build

.PHONY: format
format:  ## Run cargo fmt (workspace)
	cargo fmt

.PHONY: clippy
clippy:  ## Run cargo clippy (workspace)
	cargo clippy

.PHONY: lint
lint: format clippy  ## Run format and clippy

.PHONY: dev-lint
dev-lint: format  ## Run format and clippy --fix to auto-fix errors
	cargo clippy --fix --allow-dirty --allow-staged

.PHONY: test
test:  ## Run Rust tests (workspace)
	cargo test

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
