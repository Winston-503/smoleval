# smoleval-cli

CLI for [smoleval](https://crates.io/crates/smoleval), a minimal evaluation framework for AI agents.

Run evaluations against HTTP agent endpoints and generate reports in multiple formats.

## Installation

```bash
cargo install smoleval-cli
```

## Usage

```bash
smoleval --dataset eval.yaml --agent http://localhost:3825
```

### Options

| Flag            | Description                                    | Default |
|-----------------|------------------------------------------------|---------|
| `--dataset`     | Path to the YAML dataset file                  | —       |
| `--agent`       | URL of the HTTP agent endpoint                 | —       |
| `--format`      | Output format: `text`, `json`, or `junit`      | `text`  |
| `--concurrency` | Number of parallel test runs                   | `1`     |
| `--fail-fast`   | Stop on first failure                          | `false` |
| `--threshold`   | Minimum mean score (0.0–1.0), exits 1 if below | —       |
| `--timeout`     | HTTP request timeout in seconds                | `60`    |
| `--output`      | Write report to file instead of stdout         | —       |
| `--quiet`       | Suppress per-test output                       | `false` |

### Output formats

- **text** — human-readable summary with per-test details
- **json** — structured output for programmatic consumption
- **junit** — XML format for CI/CD integration
