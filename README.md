---
title: Fake News Verifier
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_file: server/app.py
pinned: false
---

# 🤖 Agentic Fake News Verifier

An **evaluation environment for LLM agents**, built for the **Meta PyTorch OpenEnv Hackathon**. It tests how effectively an agent verifies news headlines through a multi-step reasoning workflow: gather evidence first, then commit to a verdict.

It includes an OpenEnv-compliant FastAPI server, a shaped reward function, per-task graders, and a baseline inference script that runs an LLM (Qwen2.5-72B-Instruct by default) against every task with structured logs.

> The YAML block at the top of this file is the Hugging Face Spaces configuration and is required for deployment.

## Environment Description

The Fake News Verifier simulates a fact-checking workflow where an AI agent must:

1. **Analyze** a news headline presented as an observation
2. **Search** for supporting/contradicting evidence
3. **Verify** the headline as True or False based on evidence

The environment rewards agents for gathering evidence before making judgments, encouraging thorough investigation over hasty conclusions.

## Action Space

The agent can take two types of actions:

| Action | Format | Description |
|--------|--------|-------------|
| `search` | `search(keywords)` | Search for evidence related to the headline |
| `verify` | `verify(true/false)` | Submit final verdict on the headline's truthfulness |

### Action Schema (Pydantic)
```python
class NewsAction(BaseModel):
    action_type: str   # "search" or "verify"
    query_or_label: str  # search keywords or "true"/"false"
```

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `headline` | `str` | The news headline to verify |
| `evidence` | `str` | Currently available evidence (updates after search) |
| `steps_left` | `int` | Remaining steps in the episode |

## Tasks

| Task ID | Difficulty | Headline | Expected Label |
|---------|-----------|----------|----------------|
| `task-1` | Easy | "NASA confirms Moon is Swiss Cheese." | `false` |
| `task-2` | Medium | "New policy: 1000 units for all tomorrow." | `true` |
| `task-3` | Hard | "Coffee leads to 20% IQ increase." | `false` |

## Reward Structure

| Action | Reward | Condition |
|--------|--------|-----------|
| Search (first) | `0.15` | First search action provides new evidence |
| Search (repeated) | `0.08` | Subsequent searches (diminishing returns) |
| Verify (correct) | `0.80` | Correct true/false label |
| Verify (incorrect) | `0.05` | Wrong label |
| Any other step | `0.05` | Base reward |

All rewards are clamped to the range `(0.01, 0.99)` — never exactly 0 or 1.

## Reward analysis & known limitations

After the hackathon I audited the environment the way I would audit any evaluation: by asking which agent strategies the reward actually favours. Measured cumulative reward per episode on `task-1`:

| Agent strategy | Episode reward |
|---|---|
| Verify correctly, no search | 0.80 |
| Search once, then verify correctly | 0.95 |
| Search 4×, then verify correctly | 1.19 (score clamped to 0.99) |
| Search 5×, **never answer** | 0.47 |
| Search 4×, then verify **incorrectly** | 0.44 |
| Search once, then verify **incorrectly** | 0.20 |

**Findings**

1. **Reward hacking via stalling.** Repeated searches return no new evidence but still earn 0.08 each, so padding the episode with searches beats the efficient path (1.19 vs 0.95).
2. **Not answering is rewarded.** An agent that never verifies scores 0.47, more than double an agent that searches once and answers wrongly (0.20). The environment rewards activity, not a correct decision.
3. **Wrong verdicts aren't penalised.** An incorrect label earns the same 0.05 as a no-op step.
4. **Difficulty labels aren't calibrated.** All three tasks use the same mechanics, a single fixed search result, and a binary label; "hard" isn't measurably harder for a model.
5. **The grader rewards keywords.** `fake_news_grader` gives 0.38 for a required keyword (e.g. `"study"`) appearing in the evidence text, which can be matched without the agent understanding the evidence.

**Fixes I would make in the next version**

- Give repeated searches with no new information a small *negative* reward, and add a per-step cost.
- Give episodes that end without a verdict a terminal reward below that of any answered episode, and give wrong verdicts a clear penalty.
- Score the final verdict separately from process rewards, so the headline metric is accuracy.
- Expand to a larger task set with retrieval over multiple, partly conflicting sources, and calibrate difficulty from measured model pass rates.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Reset environment (optional `task_id` in body) |
| `/step` | POST | Execute agent action |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all tasks with grader info |
| `/grade` | POST | Grade agent output for a specific task |

## Setup & Local Testing

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Server
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference
```bash
export HF_TOKEN="your-token-here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Docker
```bash
docker build -t fake-news-verifier .
docker run -p 7860:7860 fake-news-verifier
```

## Project Structure

```
├── server/
│   ├── __init__.py      # Package init
│   ├── app.py           # FastAPI server with OpenEnv endpoints
│   ├── logic.py         # Core environment engine + reward logic
│   └── tasks.py         # Task definitions + grader functions
├── inference.py          # Baseline inference script (runs all 3 tasks)
├── openenv.yaml          # OpenEnv environment specification
├── Dockerfile            # Container config for HF Spaces
├── pyproject.toml        # Project metadata
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## License

MIT

## Author

**Nihal Yadav**, AI Trainer (LLM evaluation & software engineering) · M.Tech AI & ML, IIT Delhi · [GitHub](https://github.com/NIHAL670) · [LinkedIn](https://linkedin.com/in/nihal-yadav-5131582b8)
