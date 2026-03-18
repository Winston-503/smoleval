# smoleval-example

Example demonstrating how to use [smoleval](https://crates.io/crates/smoleval) as a library.

Shows how to:

- Implement the `Agent` trait with a mock echo agent
- Define and register custom checks, including LLM-as-a-judge implemented with [Rig](https://github.com/0xPlaygrounds/rig)
- Load an evaluation dataset from YAML and run evaluations

## Running

```bash
export OPENAI_API_KEY=sk-...  # could be fake one, but without the key preflight check will fail

make example-smoleval
```
