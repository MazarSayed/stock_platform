# Stock Trading Platform Assistant

A LangGraph-based AI assistant for stock trading platforms with RAG capabilities for FAQ, market analysis, and trading operations.

## Features

- **Multi-Agent System**: FAQ agent, Task agent, and Market Insights agent
- **RAG Integration**: Document-based Q&A using vector embeddings
- **FastAPI Backend**: RESTful API for chat interactions
- **Streamlit Frontend**: Web interface for testing and interaction
- **Langfuse Observability**: Automatic tracing and monitoring of all agent interactions
- **Security Guardrails**: Multi-layer protection against prompt injection, sensitive data exposure, and invalid operations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and fill in your actual API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key  # Optional
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key  # Optional
   LANGFUSE_HOST=https://cloud.langfuse.com  # Optional
   ```

3. Build the vectorstore:
```bash
python scripts/build_vectorstore.py
```

## Usage

**Start the FastAPI server:**
```bash
uvicorn main:app --reload
```

**Start the Streamlit frontend:**
```bash
streamlit run app.py
```

Access the Streamlit app at `http://localhost:8501` and the API at `http://localhost:8000`.

## Evaluation

The platform includes an automated evaluation system to test agent performance against ground truth test cases.

### Running Evaluation

Run the evaluation script:
```bash
python scripts/run_evaluation.py
```

### Evaluation Metrics

The system evaluates:
- **Relevance Score**: How relevant the answer is to the question (0-1 scale)
- **Accuracy Score**: How accurate the answer is compared to ground truth (0-1 scale)
- **Agent Match**: Whether the correct agent handled the query
- **Tools Match**: Whether the expected tools were used
- **Latency**: Response time in milliseconds

### Ground Truth Data

Test cases are defined in `data/evaluation/ground_truth.yaml`. Each test case includes:
- Question
- Expected answer
- Expected agent
- Expected tools

### Results

Evaluation results are saved to `data/evaluation/results/` as JSON files containing:
- Individual test metrics
- Aggregate statistics (pass rate, average scores, tool usage)
- Summary reports

Results are also logged to Langfuse for observability and analysis.

## Langfuse Integration

The platform integrates with [Langfuse](https://langfuse.com) for observability, tracing, and monitoring of agent interactions.

### Features

- **Automatic Tracing**: All API requests and agent interactions are automatically traced
- **Session Tracking**: Each chat session is tracked with a unique session ID
- **Tool Call Monitoring**: All tool invocations (RAG, trading, web search) are logged
- **Evaluation Scoring**: Evaluation metrics are automatically scored in Langfuse
- **Performance Metrics**: Latency and response times are tracked

### Setup

1. **Get Langfuse credentials**:
   - Sign up at [Langfuse Cloud](https://cloud.langfuse.com) or deploy self-hosted
   - Get your `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`

2. **Configure environment variables**:
   ```env
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, defaults to cloud
   ```

3. **Usage**:
   - Traces are automatically created for all API requests
   - View traces, scores, and metrics in the Langfuse dashboard
   - Evaluation runs create dedicated traces with scores

### What's Tracked

- **Traces**: Complete request/response cycles with full conversation context
- **Spans**: Individual agent and tool invocations
- **Scores**: Evaluation metrics (relevance, accuracy, agent/tool matching)
- **Metadata**: Session IDs, agent names, tool usage patterns

### Optional

Langfuse integration is optional. If credentials are not provided, the system will continue to work without observability features.

## Architecture

The platform uses a LangGraph-based multi-agent architecture with a supervisor pattern:

![LangGraph Architecture](langgraph_architecture.png)

**Graph Flow:**
- **START** → Routes to **Supervisor**
- **Supervisor** → Conditionally routes to one of three agents:
  - `faq_agent` - Handles platform questions using RAG
  - `task_agent` - Executes trading operations
  - `market_insights_agent` - Provides market analysis with RAG + web search
- **Agents** → Return to **Supervisor**
- **Supervisor** → Routes to **END** when complete

The supervisor uses LLM-based routing to determine which agent should handle each query based on user intent.

## Security Guardrails

The platform includes comprehensive security guardrails to protect against malicious inputs, sensitive data exposure, and invalid trading operations.

### Input Guardrails

Validates and sanitizes all user inputs before processing:

- **Prompt Injection Detection**: Blocks common injection patterns like "ignore previous instructions", system delimiters, and role-playing attempts
- **Off-Topic Filtering**: Prevents queries unrelated to stock trading (hacking, malware, illegal activities)
- **Input Sanitization**: Automatically removes suspicious patterns and injection delimiters

### Output Guardrails

Validates and sanitizes all agent responses before returning to users:

- **Sensitive Data Detection**: Identifies and redacts API keys, tokens, credit cards, SSNs, and passwords
- **Domain Boundary Enforcement**: Blocks responses containing off-topic content or harmful code patterns
- **Automatic Sanitization**: Redacts sensitive patterns with `[REDACTED-*]` placeholders while preserving response structure

### Tool Guardrails

Validates all trading tool invocations:

- **Symbol Validation**: Ensures stock symbols are properly formatted (1-5 alphanumeric characters)
- **Quantity Limits**: Enforces minimum (1) and maximum (10,000) shares/contracts per order
- **Order Type Validation**: Only allows "market" or "limit" order types
- **Session Limits**: Tracks and limits orders per session (max 10 orders per session)

### Implementation

Guardrails are automatically applied at multiple layers:
- **Input validation** happens in the API endpoint before processing
- **Tool validation** happens within each trading tool before execution
- **Output validation** happens before responses are returned to users

All guardrail actions are logged for monitoring and analysis.

## Project Structure

- `main.py` - FastAPI backend server
- `app.py` - Streamlit frontend
- `graph/` - LangGraph agent orchestration
- `nodes/` - Agent node implementations
- `rag/` - RAG components (retriever, embedder, vectorstore)
- `tools/` - Agent tools (FAQ, market analysis, trading)
- `guardrails/` - Security guardrails (input, output, tool validation)
- `evaluation/` - Evaluation system (auto evaluator, metrics tracker)
- `data/` - Documents, vectorstore, and evaluation ground truth
- `scripts/` - Utility scripts for setup and evaluation