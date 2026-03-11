RS_DIR := crates/smoleval

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

.PHONY: doc
doc:  ## Run cargo doc for smoleval
	cd $(RS_DIR) && cargo doc --open

.PHONY: publish-dry-run
publish-dry-run:  ## Publish smoleval in dry run
	cd $(RS_DIR) && cargo publish --dry-run

.PHONY: publish
publish:  ## Publish smoleval to crates.io
	cd $(RS_DIR) && cargo publish
