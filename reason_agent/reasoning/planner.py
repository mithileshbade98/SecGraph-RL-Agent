"""
Reasoning planner - generates step-by-step plans for query resolution.

Uses LLM (Ollama local) with deterministic fallback heuristic.
Emits reasoning trace with tool calls, evidence, and rationales.
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from loguru import logger
from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.tools.executors import ExecutorRegistry


class Step:
    """A single reasoning step."""

    def __init__(
        self,
        step_id: int,
        thought: str,
        tool: str,
        parameters: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
    ):
        self.step_id = step_id
        self.thought = thought
        self.tool = tool
        self.parameters = parameters
        self.result = result
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'thought': self.thought,
            'tool': self.tool,
            'parameters': self.parameters,
            'result': self.result,
            'timestamp': self.timestamp.isoformat(),
        }


class ReasoningPlanner:
    """Plan and execute reasoning steps."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_router: ToolRouter,
        executor_registry: ExecutorRegistry,
        max_steps: int = 10,
        use_llm: bool = False,
    ):
        """
        Initialize planner.

        Args:
            tool_registry: Tool registry
            tool_router: Tool router
            executor_registry: Executor registry
            max_steps: Maximum planning steps
            use_llm: Use LLM for planning (vs deterministic heuristic)
        """
        self.tool_registry = tool_registry
        self.tool_router = tool_router
        self.executor_registry = executor_registry
        self.max_steps = max_steps
        self.use_llm = use_llm

    def plan_and_execute(
        self,
        query: str,
        enable_trace: bool = True,
    ) -> Dict[str, Any]:
        """
        Plan steps and execute them.

        Returns:
            {
                'query': str,
                'steps': List[Step],
                'final_result': Any,
                'success': bool,
                'reasoning_trace': List[Dict],
            }
        """
        logger.info(f"Planning for query: {query}")

        steps = []

        if self.use_llm:
            # LLM-based planning (would call Ollama API)
            steps = self._llm_plan(query)
        else:
            # Deterministic heuristic planning
            steps = self._heuristic_plan(query)

        # Execute steps
        results = []
        for step in steps:
            logger.info(f"Step {step.step_id}: {step.thought}")
            logger.info(f"  Tool: {step.tool} with params: {step.parameters}")

            # Get tool from registry
            tool = self.tool_registry.get_tool(step.tool)
            if not tool:
                logger.error(f"Unknown tool: {step.tool}")
                step.result = {'success': False, 'error': f'Unknown tool: {step.tool}'}
                continue

            # Execute tool
            result = self.executor_registry.execute(tool.executor, step.parameters)
            step.result = result
            results.append(result)

            logger.info(f"  Result: {result.get('success', False)}")

        # Determine overall success
        success = all(r.get('success', False) for r in results)

        # Compile trace
        trace = {
            'query': query,
            'steps': [step.to_dict() for step in steps],
            'final_result': results[-1] if results else None,
            'success': success,
            'num_steps': len(steps),
            'timestamp': datetime.now().isoformat(),
        }

        return trace

    def _llm_plan(self, query: str) -> List[Step]:
        """LLM-based planning (stub - would call Ollama)."""
        # In production, this would:
        # 1. Format query with tool cards
        # 2. Call LLM API (Ollama/OpenAI/Azure)
        # 3. Parse LLM response to extract steps

        # For now, fallback to heuristic
        logger.warning("LLM planning not implemented, using heuristic")
        return self._heuristic_plan(query)

    def _heuristic_plan(self, query: str) -> List[Step]:
        """Deterministic heuristic planning based on query patterns."""
        query_lower = query.lower()
        steps = []
        step_id = 1

        # Pattern: multi-account detection
        if "multi-account" in query_lower or "shared device" in query_lower:
            steps.append(Step(
                step_id=step_id,
                thought="Check for shared device policy violations",
                tool="policy_checker",
                parameters={'pattern': 'multi_account_same_device', 'threshold': 0.8}
            ))
            step_id += 1

            steps.append(Step(
                step_id=step_id,
                thought="Detect account clusters sharing devices",
                tool="cluster_detector",
                parameters={'algorithm': 'louvain', 'min_cluster_size': 3}
            ))
            step_id += 1

        # Pattern: velocity/rate limiting
        elif "velocity" in query_lower or "rate limit" in query_lower:
            steps.append(Step(
                step_id=step_id,
                thought="Check velocity violations",
                tool="velocity_checker",
                parameters={'entity_id': 'user_example', 'event_type': 'login', 'max_events': 50,
                           'time_window_hours': 24}
            ))
            step_id += 1

        # Pattern: burst detection
        elif "burst" in query_lower or "storm" in query_lower:
            steps.append(Step(
                step_id=step_id,
                thought="Detect temporal bursts of activity",
                tool="burst_detector",
                parameters={'entity_type': 'account', 'event_type': 'signup',
                           'window_minutes': 60, 'threshold_multiplier': 3.0}
            ))
            step_id += 1

        # Default: use tool router to find relevant tools
        else:
            routed_tools = self.tool_router.route(query, top_k=2)
            for i, (tool, score) in enumerate(routed_tools):
                steps.append(Step(
                    step_id=step_id,
                    thought=f"Use {tool.name} to address query (confidence: {score:.2f})",
                    tool=tool.name,
                    parameters=self._generate_default_parameters(tool)
                ))
                step_id += 1

        # Limit steps
        return steps[:self.max_steps]

    def _generate_default_parameters(self, tool) -> Dict[str, Any]:
        """Generate default parameters for a tool."""
        params = {}
        for param in tool.parameters:
            if param.default is not None:
                params[param.name] = param.default
            elif param.type == "string":
                params[param.name] = "example_value"
            elif param.type == "integer":
                params[param.name] = 10
            elif param.type == "float":
                params[param.name] = 0.8
        return params


def main():
    """Test planner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test reasoning planner")
    parser.add_argument("--query", type=str, required=True, help="Query to plan")
    args = parser.parse_args()

    # Set up components
    registry = ToolRegistry()
    router = ToolRouter(registry, method="rule")
    executors = ExecutorRegistry()

    planner = ReasoningPlanner(
        tool_registry=registry,
        tool_router=router,
        executor_registry=executors,
        use_llm=False,
    )

    # Plan and execute
    trace = planner.plan_and_execute(args.query)

    logger.info("\n=== Reasoning Trace ===")
    logger.info(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
