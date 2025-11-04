"""Utilities for delegating work to specialised external agents.

The module mirrors the conventions used in the Multi-Agent-Platform
project so that this service can collaborate with the Browser Agent and
IoT Agent described in the upstream repositories.  Each agent definition
contains an inline summary of its responsibilities and the HTTP
endpoints used for coordination.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


def _load_env_list(name: str, default: Iterable[str]) -> List[str]:
    """Return a normalised list of URLs read from a comma separated env var."""

    raw_value = os.getenv(name)
    if not raw_value:
        return list(default)

    values = []
    for candidate in raw_value.split(","):
        cleaned = candidate.strip()
        if cleaned:
            values.append(cleaned)
    return values or list(default)


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a specialist agent that can support this service."""

    name: str
    role_description: str
    base_urls: List[str]
    help_path: str
    timeout: float
    conversation_path: Optional[str] = None

    def iter_urls(self, path: Optional[str] = None) -> Iterator[str]:
        """Yield absolute URLs combining the base URL and a relative path."""

        target_path = path or self.help_path
        for base in self.base_urls:
            yield urljoin(base.rstrip("/"), target_path)


class AgentDispatchError(RuntimeError):
    """Raised when the service cannot forward a request to a peer agent."""

    def __init__(self, message: str, *, details: Optional[Mapping[str, str]] = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class AgentDelegator:
    """Delegate questions to Browser/IoT agents with lightweight heuristics."""

    #: Browser agent: automates Chrome using browser-use.  The relay endpoint
    #: receives natural-language instructions and executes them without touching
    #: the agent's primary chat log.
    _BROWSER_AGENT = AgentConfig(
        name="browser",
        role_description=(
            "Web リサーチやサイト操作を担当するブラウザ自動化エージェント。"
            "`/api/agent-relay` で単発タスク、`/api/check-conversation-history` で"
            "会話履歴を解析し必要なブラウザ操作を判断します。"
        ),
        base_urls=_load_env_list(
            "BROWSER_AGENT_BASE_URLS",
            ["http://browser-agent:5005"],
        ),
        help_path="/api/agent-relay",
        timeout=float(os.getenv("BROWSER_AGENT_TIMEOUT", "120")),
        conversation_path="/api/check-conversation-history",
    )

    #: IoT agent: manages household devices and returns operational guidance.
    #: `/api/agents/respond` handles ad-hoc instructions while
    #: `/api/conversations/review` inspects recent dialogue and proposes device
    #: actions.
    _IOT_AGENT = AgentConfig(
        name="iot",
        role_description=(
            "家電やセンサーの状態確認・制御を担う IoT エージェント。"
            "`/api/agents/respond` で操作リクエスト、"
            "`/api/conversations/review` で会話レビューを受け付けます。"
        ),
        base_urls=_load_env_list(
            "IOT_AGENT_BASE_URLS",
            ["https://iot-agent.project-kk.com"],
        ),
        help_path="/api/agents/respond",
        timeout=float(os.getenv("IOT_AGENT_TIMEOUT", "30")),
        conversation_path="/api/conversations/review",
    )

    _ALIAS_MAP: Mapping[str, str] = {
        "browser": "browser",
        "web": "browser",
        "navigator": "browser",
        "web_agent": "browser",
        "browser_agent": "browser",
        "iot": "iot",
        "device": "iot",
        "iot_agent": "iot",
    }

    _IOT_KEYWORDS = (
        "デバイス",
        "家電",
        "温度",
        "電源",
        "照明",
        "エアコン",
        "掃除機",
        "IoT",
        "センサー",
    )
    _BROWSER_KEYWORDS = (
        "ウェブ",
        "Web",
        "ブラウザ",
        "検索",
        "サイト",
        "ページ",
        "スクレイプ",
        "スクリーンショット",
        "調査",
    )

    def __init__(self) -> None:
        self._agents: Mapping[str, AgentConfig] = {
            "browser": self._BROWSER_AGENT,
            "iot": self._IOT_AGENT,
        }

    def get_agent(self, name_or_alias: str) -> AgentConfig:
        """Return an agent configuration by canonical name or alias."""

        key = name_or_alias.lower().strip()
        canonical = self._ALIAS_MAP.get(key, key)
        agent = self._agents.get(canonical)
        if not agent:
            raise AgentDispatchError(f"Unknown agent: {name_or_alias}")
        return agent

    def choose_agent(self, request_text: str, hint: Optional[str] = None) -> AgentConfig:
        """Pick the most suitable agent based on hints and keyword heuristics."""

        if hint:
            try:
                return self.get_agent(hint)
            except AgentDispatchError as exc:
                logger.warning("Invalid agent hint '%s': %s", hint, exc)

        lowered = request_text.lower()

        if any(keyword.lower() in lowered for keyword in self._IOT_KEYWORDS):
            return self._IOT_AGENT

        if any(keyword.lower() in lowered for keyword in self._BROWSER_KEYWORDS):
            return self._BROWSER_AGENT

        # Fall back to the browser agent for general research tasks.
        return self._BROWSER_AGENT

    def delegate_help_request(
        self,
        request_text: str,
        *,
        hint: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
        supplemental_context: Optional[str] = None,
    ) -> Dict[str, object]:
        """Send a help request to the selected agent and return its response."""

        agent = self.choose_agent(request_text, hint=hint)
        payload = self._build_payload(agent, request_text, metadata, supplemental_context)
        response_body = self._post_json(agent, agent.help_path, payload)

        return {
            "agent": agent.name,
            "agent_role": agent.role_description,
            "request_payload": payload,
            "agent_response": response_body,
        }

    def forward_conversation(
        self,
        conversation_history: Iterable[Mapping[str, str]],
        *,
        agent: Optional[str] = None,
    ) -> Dict[str, object]:
        """Forward conversation history to the configured agent review endpoint."""

        if agent:
            config = self.get_agent(agent)
        else:
            config = self._BROWSER_AGENT

        if not config.conversation_path:
            raise AgentDispatchError(
                f"Agent '{config.name}' does not expose a conversation review endpoint."
            )

        normalised_history: List[Dict[str, str]] = []
        for entry in conversation_history:
            role = entry.get("role") if isinstance(entry, Mapping) else None
            content = entry.get("content") if isinstance(entry, Mapping) else None
            if isinstance(role, str) and isinstance(content, str):
                normalised_history.append({"role": role, "content": content})

        payload = {"conversation_history": normalised_history}
        response_body = self._post_json(config, config.conversation_path, payload)
        return {
            "agent": config.name,
            "agent_role": config.role_description,
            "request_payload": payload,
            "agent_response": response_body,
        }

    def _build_payload(
        self,
        agent: AgentConfig,
        request_text: str,
        metadata: Optional[Mapping[str, object]],
        supplemental_context: Optional[str],
    ) -> Dict[str, object]:
        if agent.name == "browser":
            prompt_parts = [request_text.strip()]
            if supplemental_context:
                prompt_parts.append(f"追加コンテキスト:\n{supplemental_context.strip()}")
            prompt = "\n\n".join(part for part in prompt_parts if part)
            payload: MutableMapping[str, object] = {"prompt": prompt}
            if metadata:
                payload["metadata"] = dict(metadata)
            return dict(payload)

        if agent.name == "iot":
            payload = {"request": request_text.strip()}
            if metadata:
                payload["metadata"] = dict(metadata)
            if supplemental_context:
                payload["context"] = supplemental_context.strip()
            return payload

        raise AgentDispatchError(f"Payload builder missing for agent '{agent.name}'.")

    def _post_json(
        self,
        agent: AgentConfig,
        path: str,
        payload: Mapping[str, object],
    ) -> Dict[str, object]:
        errors: Dict[str, str] = {}
        for url in agent.iter_urls(path):
            try:
                response = requests.post(url, json=payload, timeout=agent.timeout)
            except requests.RequestException as exc:
                logger.warning("Failed to reach %s: %s", url, exc)
                errors[url] = str(exc)
                continue

            if response.status_code >= 500:
                errors[url] = f"{response.status_code}: {response.text.strip()}"
                continue

            try:
                body = response.json()
            except json.JSONDecodeError as exc:
                errors[url] = f"Invalid JSON response: {exc}"
                continue

            if response.status_code >= 400:
                message = body.get("error") if isinstance(body, Mapping) else None
                errors[url] = message or f"{response.status_code}: {response.text.strip()}"
                continue

            if not isinstance(body, Mapping):
                errors[url] = "Response must be a JSON object"
                continue

            return dict(body)

        raise AgentDispatchError(
            f"Failed to contact {agent.name} agent at any configured endpoint.",
            details=errors,
        )


__all__ = ["AgentDelegator", "AgentDispatchError", "AgentConfig"]
