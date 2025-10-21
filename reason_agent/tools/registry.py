"""
Tool registry for managing available agent tools.

Loads tool definitions from tools.yaml and provides lookup/search functionality.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ToolParameter:
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """Tool definition."""
    name: str
    description: str
    category: str
    parameters: List[ToolParameter]
    executor: str
    examples: List[str] = field(default_factory=list)
    production_only: bool = False

    def to_embedding_text(self) -> str:
        """Convert tool to text for embedding."""
        param_text = ", ".join([
            f"{p.name} ({p.type}): {p.description}"
            for p in self.parameters
        ])
        examples_text = "\n".join(self.examples)
        return f"{self.name}\n{self.description}\nParameters: {param_text}\nExamples:\n{examples_text}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'parameters': [
                {
                    'name': p.name,
                    'type': p.type,
                    'description': p.description,
                    'required': p.required,
                    'default': p.default,
                    'enum': p.enum,
                }
                for p in self.parameters
            ],
            'executor': self.executor,
            'examples': self.examples,
            'production_only': self.production_only,
        }


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize tool registry.

        Args:
            config_path: Path to tools.yaml (default: configs/tools.yaml)
        """
        if config_path is None:
            config_path = Path("configs/tools.yaml")

        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}

        self.load_tools(config_path)

    def load_tools(self, config_path: Path):
        """Load tools from YAML configuration."""
        if not config_path.exists():
            logger.warning(f"Tool config not found: {config_path}")
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        tools_config = config.get('tools', [])

        for tool_def in tools_config:
            # Parse parameters
            parameters = []
            for param in tool_def.get('parameters', []):
                parameters.append(ToolParameter(
                    name=param['name'],
                    type=param['type'],
                    description=param.get('description', ''),
                    required=param.get('required', True),
                    default=param.get('default'),
                    enum=param.get('enum'),
                ))

            # Create tool
            tool = Tool(
                name=tool_def['name'],
                description=tool_def['description'],
                category=tool_def['category'],
                parameters=parameters,
                executor=tool_def['executor'],
                examples=tool_def.get('examples', []),
                production_only=tool_def.get('production_only', False),
            )

            self.tools[tool.name] = tool

            # Add to category index
            if tool.category not in self.categories:
                self.categories[tool.category] = []
            self.categories[tool.category].append(tool.name)

        logger.info(f"Loaded {len(self.tools)} tools from {config_path}")
        logger.info(f"Categories: {list(self.categories.keys())}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category."""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def get_all_tools(self) -> List[Tool]:
        """Get all tools."""
        return list(self.tools.values())

    def get_tool_names(self) -> List[str]:
        """Get all tool names."""
        return list(self.tools.keys())

    def search_tools(self, query: str, limit: int = 5) -> List[Tool]:
        """
        Simple keyword search over tools.

        For semantic search, use ToolRouter instead.
        """
        query_lower = query.lower()
        scored_tools = []

        for tool in self.tools.values():
            score = 0
            # Check name
            if query_lower in tool.name.lower():
                score += 10
            # Check description
            if query_lower in tool.description.lower():
                score += 5
            # Check examples
            for example in tool.examples:
                if query_lower in example.lower():
                    score += 3

            if score > 0:
                scored_tools.append((score, tool))

        # Sort by score
        scored_tools.sort(reverse=True, key=lambda x: x[0])

        return [tool for _, tool in scored_tools[:limit]]

    def get_embeddings_data(self) -> List[Dict[str, Any]]:
        """
        Get tool data formatted for embedding.

        Returns list of dicts with 'id', 'text', and 'metadata'.
        """
        data = []
        for tool in self.tools.values():
            data.append({
                'id': tool.name,
                'text': tool.to_embedding_text(),
                'metadata': tool.to_dict(),
            })
        return data

    def validate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a tool call.

        Returns (is_valid, error_message)
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return False, f"Unknown tool: {tool_name}"

        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in parameters:
                return False, f"Missing required parameter: {param.name}"

        # Check enum constraints
        for param in tool.parameters:
            if param.enum and param.name in parameters:
                if parameters[param.name] not in param.enum:
                    return False, f"Invalid value for {param.name}: expected one of {param.enum}"

        return True, None


def main():
    """Test tool registry."""
    registry = ToolRegistry()

    logger.info("\nAll tools:")
    for tool in registry.get_all_tools():
        logger.info(f"  {tool.name} ({tool.category}): {tool.description}")

    logger.info("\nSearch for 'policy':")
    results = registry.search_tools("policy")
    for tool in results:
        logger.info(f"  {tool.name}: {tool.description}")

    logger.info("\nTools by category 'verification':")
    verification_tools = registry.get_tools_by_category("verification")
    for tool in verification_tools:
        logger.info(f"  {tool.name}")

    logger.info("\nEmbedding data sample:")
    embedding_data = registry.get_embeddings_data()
    if embedding_data:
        logger.info(f"  {embedding_data[0]['id']}")
        logger.info(f"  {embedding_data[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
