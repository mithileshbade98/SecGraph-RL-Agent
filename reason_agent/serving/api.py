"""
FastAPI serving layer for SecGraph-RL Agent.

Endpoints:
- POST /query: Run agent query
- GET /trace/{trace_id}: Get reasoning trace
- GET /healthz: Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.tools.executors import ExecutorRegistry
from reason_agent.reasoning.planner import ReasoningPlanner
from reason_agent.reasoning.trace_recorder import TraceRecorder
from reason_agent.reasoning.verifiers.policy_verifier import PolicyVerifier
from reason_agent.rl.rewards import RewardComputer

app = FastAPI(title="SecGraph-RL Agent API", version="0.1.0")

# Initialize components
registry = ToolRegistry()
router = ToolRouter(registry, method="rule")
executors = ExecutorRegistry()
planner = ReasoningPlanner(registry, router, executors, use_llm=False)
trace_recorder = TraceRecorder()
policy_verifier = PolicyVerifier()
reward_computer = RewardComputer({
    'process_weight': 0.3,
    'final_weight': 0.7,
    'penalty_long_trace': -0.1,
    'penalty_redundant_tool': -0.05,
    'bonus_early_success': 0.2,
})


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    max_steps: Optional[int] = 10
    enable_verification: Optional[bool] = True


class QueryResponse(BaseModel):
    """Query response model."""
    trace_id: str
    query: str
    success: bool
    final_result: Dict[str, Any]
    num_steps: int
    rewards: Dict[str, float]
    verification: List[Dict[str, Any]]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute agent query.

    Returns reasoning trace with verifiable rewards.
    """
    logger.info(f"Received query: {request.query}")

    try:
        # Plan and execute
        trace = planner.plan_and_execute(request.query)

        # Verify
        verification_results = []
        if request.enable_verification:
            # Run policy verification
            policy_result = policy_verifier.verify(
                "no_shared_device_abuse",
                trace,
            )
            verification_results.append({
                'verifier': 'policy_verifier',
                **policy_result,
            })

        # Compute rewards
        rewards = reward_computer.compute_reward(trace, verification_results)

        # Record trace
        trace_id = trace_recorder.record_trace(
            query=request.query,
            retrieval_results=[],
            steps=trace.get('steps', []),
            final_result=trace.get('final_result', {}),
            rewards=rewards,
            verification_results=verification_results,
        )

        return QueryResponse(
            trace_id=trace_id,
            query=request.query,
            success=trace.get('success', False),
            final_result=trace.get('final_result', {}),
            num_steps=trace.get('num_steps', 0),
            rewards=rewards,
            verification=verification_results,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Get reasoning trace by ID."""
    trace = trace_recorder.load_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    return trace


@app.get("/healthz")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }


@app.get("/stats")
async def stats():
    """Get system statistics."""
    return {
        "tools": len(registry.get_all_tools()),
        "traces": trace_recorder.get_trace_stats(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
