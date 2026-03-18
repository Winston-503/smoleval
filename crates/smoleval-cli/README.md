# smoleval-cli

CLI for [smoleval](https://crates.io/crates/smoleval).

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

| Flag            | Description                                      | Default |
|-----------------|--------------------------------------------------|---------|
| `--dataset`     | Path to the YAML dataset file                    | —       |
| `--agent`       | URL of the HTTP agent endpoint                   | —       |
| `--concurrency` | Number of parallel test runs                     | `1`     |
| `--fail-fast`   | Stop on first failure                            | `false` |
| `--threshold`   | Minimum mean score (0.0 - 1.0), exits 1 if below | —       |
| `--timeout`     | HTTP request timeout in seconds                  | `60`    |
| `--output`      | Write report to file                             | —       |
| `--quiet`       | Suppress per-test output                         | `false` |

### Output formats

The output format is inferred from the `--output` file extension:

- `.json` — structured output for programmatic consumption
- `.xml` — JUnit XML format for CI/CD integration
- any other extension — human-readable text summary

Stdout always uses the text format.
