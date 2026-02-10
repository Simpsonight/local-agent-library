# Local Agent Library (LAL)

## Overview
CLI framework for manually orchestrating specialized AI agents. Agents are discovered by their folder structure under `agents/`.

## Architecture
- API keys are loaded from a `.env` file at startup via `python-dotenv`; `litellm` picks them up from the environment automatically
- `main.py` – Interactive CLI loop (agent → template → variables → LLM → result)
- `core/engine.py` – `Agent` class (loads system.txt, config.yaml, .j2 templates), `SessionCache` singleton
- `core/utils.py` – Web scraper, file I/O
- `agents/<name>/` – Each agent folder contains `system.txt`, optional `config.yaml`, and `.j2` templates

## Key Commands
```bash
pip install -r requirements.txt   # Install dependencies
python main.py                    # Run the interactive CLI
```

## Agent Structure
```
agents/<agent_name>/
├── system.txt      # Required – system prompt for the LLM
├── config.yaml     # Optional – model, temperature, max_tokens
└── *.j2            # Jinja2 templates with {{ variable }} placeholders
```

## Variable Input Commands
- `/ctx` – inject last LLM output
- `/ctx <alias>` – inject a named cached result
- `/url <url>` – scrape web page content
- `/url <user:pass> <url>` – scrape with HTTP Basic Auth
- `/file <path>` – load file content
- `/txt <text>` – literal plain text (bypass command parsing)

## Dependencies
litellm, jinja2, requests, beautifulsoup4, pyyaml, python-dotenv
