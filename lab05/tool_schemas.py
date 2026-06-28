from __future__ import annotations

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (Wikipedia) and return a short summary of the top result.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Describe the content of an image given its local file path.",
            "parameters": {
                "type": "object",
                "properties": {"image_path": {"type": "string", "description": "Local path to the image file"}},
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Evaluate a basic math expression, e.g. '2 + 2 * 3'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local_knowledge",
            "description": "Search a local knowledge base (JSON) of facts for an answer.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Question or keywords to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
]
