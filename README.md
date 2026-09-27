# Aegis – Financial Risk Orchestrator

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (for full stack)
- Plaid sandbox credentials
- OpenAI API key

### 2. One-Click Docker Deployment

```bash
cd aegis
cp .env.example .env
# Edit .env with your API keys
docker-compose up --build
```

**Services:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

---

### 3. Local Development (No Docker)

**Backend:**
```bash
cd backend
cp .env.example .env
# Edit .env

python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev   # http://localhost:3000
```

---

## Architecture

```
User Input (Natural Language)
        │
        ▼
┌─────────────────────────────────────────┐
│  Node 1: Semantic Parser (LLM)          │
│  → Outputs ONLY strict Pydantic JSON    │
│  → Zero mathematical computations       │
└─────────────────┬───────────────────────┘
                  │ FinancialAction JSON
                  ▼
┌─────────────────────────────────────────┐
│  Node 2: Deterministic Solver           │
│  → calculate_cash_flow()                │
│  → simulate_shock()                     │
│  → monte_carlo_cashflow()               │
│  → GRPO optimization                    │
└─────────────────┬───────────────────────┘
                  │ Math Results
                  ▼
┌─────────────────────────────────────────┐
│  Node 3: HITL Gate (State-changing?)    │
│  → interrupt() → ActionQueue → Pause    │
│  → /api/agent/resume  ← Human Decision  │
└─────────────────────────────────────────┘
```

## Core Principles

1. **Zero LLM Math** – The LLM extracts intent only. All arithmetic via NumPy/Pandas.
2. **Human-In-The-Loop** – No financial state change without explicit human approval.
3. **Graceful Degradation** – Circuit breakers on all external APIs (Plaid, LLM). Falls back to last PostgreSQL state.
4. **Immutable Ledger** – Transactions are append-only. Never updated.
5. **GRPO Optimization** – 8 candidate allocation policies evaluated via Monte Carlo, best one proposed.

## File Structure

```
aegis/
├── backend/
│   ├── main.py                     # FastAPI entry point
│   ├── api/
│   │   ├── crud.py                 # CRUD endpoints (Users, Accounts, Transactions)
│   │   ├── ingestion.py            # Plaid sync + circuit breaker
│   │   └── hitl_queue.py          # Agent chat + HITL resume
│   ├── core/
│   │   ├── config.py               # Settings via pydantic-settings
│   │   └── checkpointer.py        # LangGraph PostgresSaver
│   ├── database/
│   │   ├── models.py              # SQLAlchemy ORM (User, Account, Transaction, ActionQueue)
│   │   └── vector.py              # Qdrant client
│   ├── environment/               # THE DETERMINISTIC HARNESS
│   │   ├── simulator.py           # Cash flow, shock, compound interest, Monte Carlo
│   │   ├── shock_generator.py     # Economic shock scenario generator
│   │   └── reward.py              # GRPO reward function (liquidity/insolvency/volatility)
│   └── agent/
│       ├── orchestrator.py        # LangGraph graph (SemanticParser→Solver→HITLGate)
│       ├── grpo_optimizer.py      # Group Relative Policy Optimization
│       └── prompts.py             # Context-engineered LLM prompts
├── frontend/                       # Next.js 15 Ghost Ledger Dashboard
│   └── src/
│       ├── app/
│       │   ├── page.tsx           # Main dashboard (Ghost Ledger + Chat + Command Queue)
│       │   └── globals.css        # Design system (dark mode, Aegis brand)
│       └── lib/api.ts             # Typed API client
├── tests/
│   └── test_math_solvers.py       # Deterministic solver unit tests
├── scripts/init.sql               # PostgreSQL initialization
├── docker-compose.yml             # Full stack orchestration
└── requirements.txt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (for LLM parser) |
| `DATABASE_URL` | PostgreSQL connection string |
| `PLAID_CLIENT_ID` | Plaid Client ID |
| `PLAID_SECRET` | Plaid Sandbox secret |
| `QDRANT_URL` | Qdrant vector DB URL |
| `GRPO_GROUP_SIZE` | Candidate policies per GRPO generation (default: 8) |
| `LLM_MODEL` | LLM model (default: gpt-4o) |

## Running Tests

```bash
cd aegis
pip install -r requirements.txt
pytest tests/ -v
```
