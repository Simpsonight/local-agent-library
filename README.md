# Local Agent Library (LAL)

CLI framework for manually orchestrating specialized AI agents. Agents are automatically discovered by their folder structure under `agents/`.

## Installation

```bash
pip install -r requirements.txt
```

Or as an editable package:

```bash
pip install -e .
```

## Usage

```bash
python main.py       # Start the interactive CLI
lal                  # Alternative after pip install -e .
```

## Configuration

API keys are loaded from a `.env` file. Template:

```bash
cp .env.example .env
# Fill in your keys
```

## Architecture

- `main.py` – Interactive CLI loop (agent → template → variables → LLM → result)
- `core/engine.py` – `Agent` class (loads system.txt, config.yaml, .j2 templates), `SessionCache`
- `core/utils.py` – Web scraper, file I/O
- `agents/<name>/` – Each agent folder contains `system.txt`, optional `config.yaml`, and `.j2` templates

## Agent Structure

```
agents/<agent_name>/
├── system.txt      # Required – system prompt for the LLM
├── config.yaml     # Optional – model, temperature, max_tokens
└── *.j2            # Jinja2 templates with {{ variable }} placeholders
```

## Variable Input

The following commands are available when filling template variables:

| Command | Description |
|---------|-------------|
| `/ctx` | Inject last LLM output |
| `/ctx <alias>` | Inject a named cached result |
| `/url <url>` | Scrape web page content |
| `/url <user:pass> <url>` | Scrape with HTTP Basic Auth |
| `/file <path>` | Load file content |
| `/txt <text>` | Plain text (bypass command parsing) |

## Dependencies

litellm, jinja2, requests, beautifulsoup4, pyyaml, python-dotenv
