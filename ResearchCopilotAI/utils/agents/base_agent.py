"""
base_agent.py
-------------
Base Agent interface for the 6-Agent Research Paper Co-Pilot Framework.
Integrated with sequential multi-key failover rotation.
"""

import json
import re
from typing import Any, Dict, Optional
from utils.llm import generate_response
from config import AGENT_CONFIGS

class BaseAgent:
    """Base class for all specialized research agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.config = AGENT_CONFIGS.get(agent_id, {
            "name": agent_id,
            "description": "Research Agent",
            "temperature": 0.2
        })
        self.name = self.config["name"]
        self.description = self.config["description"]
        self.temperature = self.config.get("temperature", 0.2)

    def call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Invoke Gemini LLM with automatic multi-key sequential failover."""
        try:
            return generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature
            )
        except Exception as e:
            return f"[Agent Error in {self.name}]: {str(e)}"

    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract structured JSON dictionary from LLM output."""
        try:
            # Match ```json ... ``` or first { ... }
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if json_match:
                raw_json = json_match.group(1).strip()
            else:
                raw_json = text.strip()

            # Find outer brackets
            start = raw_json.find("{")
            end = raw_json.rfind("}")
            if start != -1 and end != -1:
                raw_json = raw_json[start:end+1]
                return json.loads(raw_json)
        except Exception:
            pass

        return {"raw_text": text}
