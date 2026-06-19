# Student Scaffold

This `src/` folder is the student version of the lab.

- It keeps the same high-level structure
- The Python files now contain a completed offline implementation for the lab tasks
- The benchmark structure should include: standard benchmark + long-context stress benchmark
- The runtime should support these providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Suggested flow:

1. Start with `config.py`
2. Implement `memory_store.py`
3. Finish `agent_baseline.py`
4. Finish `agent_advanced.py`
5. Implement `benchmark.py`
6. Run `test_agents.py`

Datasets are available at the repo root in `data/`.
