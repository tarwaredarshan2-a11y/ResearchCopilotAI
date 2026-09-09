"""
llm.py
------
Wraps the Gemini 2.5 Flash chat model with Multi-Key Sequential Failover.

If one Google API Key reaches its rate limit or quota exhaustion (429 / ResourceExhausted),
the manager automatically fails over to the next key in sequence seamlessly.
"""

import threading
import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    GOOGLE_API_KEYS,
    GEMINI_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger("ResearchCopilotAI.LLM")

class MultiKeyRotationManager:
    """Manages sequential failover across multiple Google Gemini API keys."""

    def __init__(self, api_keys: List[str]):
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.current_index = 0
        self.lock = threading.Lock()
        self.key_failure_counts: Dict[int, int] = {i: 0 for i in range(len(self.api_keys))}

    def get_current_key(self) -> str:
        with self.lock:
            if not self.api_keys:
                return ""
            return self.api_keys[self.current_index]

    def advance_to_next_key(self, reason: str = "") -> str:
        """Switch to the next API key in sequence upon rate limiting."""
        with self.lock:
            if not self.api_keys:
                return ""
            old_idx = self.current_index
            self.key_failure_counts[old_idx] = self.key_failure_counts.get(old_idx, 0) + 1
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            new_idx = self.current_index
            print(f"[KeyManager] ⚠️ Key #{old_idx + 1} quota/rate limit reached ({reason}). Failing over to Key #{new_idx + 1} of {len(self.api_keys)} in sequence.")
            return self.api_keys[new_idx]

    def get_status(self) -> Dict[str, Any]:
        """Return status for the UI."""
        with self.lock:
            total = len(self.api_keys)
            active_num = self.current_index + 1
            masked_key = ""
            if self.api_keys:
                k = self.api_keys[self.current_index]
                masked_key = k[:6] + "..." + k[-4:] if len(k) > 10 else "******"
            return {
                "total_keys": total,
                "active_key_index": self.current_index,
                "active_key_number": active_num,
                "active_masked_key": masked_key,
                "keys_configured": total > 0
            }

# Global singleton key manager
key_manager = MultiKeyRotationManager(GOOGLE_API_KEYS)

def is_quota_or_rate_limit_error(error_msg: str) -> bool:
    """Check if an exception is due to quota exhaustion, 429, or rate limiting."""
    err = error_msg.lower()
    patterns = [
        "429", "resource_exhausted", "resourceexhausted",
        "quota", "rate limit", "too many requests", "exhausted",
        "quota exceeded", "limit exceeded", "exceeded your current quota"
    ]
    return any(p in err for p in patterns)

def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Get an LLM instance with the currently active API key."""
    current_key = key_manager.get_current_key()
    if not current_key:
        raise ValueError("No valid Google API Keys configured in .env (GOOGLE_API_KEYS).")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=current_key,
        temperature=temperature,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )

def generate_response(prompt: str, system_prompt: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    """
    Generate response from Gemini with sequential multi-key failover.
    Tries each configured API key in sequence if rate-limited.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    total_keys = max(1, len(key_manager.api_keys))
    attempts = 0
    last_exception = None

    while attempts < total_keys:
        try:
            llm = get_llm(temperature=temperature)
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)

        except Exception as exc:
            err_str = str(exc)
            last_exception = exc
            if is_quota_or_rate_limit_error(err_str) and total_keys > 1:
                key_manager.advance_to_next_key(reason=err_str[:80])
                attempts += 1
                continue
            else:
                # If non-quota error or only 1 key, raise
                raise RuntimeError(f"Gemini LLM call failed: {exc}")

    raise RuntimeError(f"All {total_keys} Google Gemini API keys in the sequential pool have been exhausted or rate-limited. Last error: {last_exception}")
