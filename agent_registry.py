"""
エージェントレジストリ - マルチエージェントシステムのエージェント情報と接続設定

このモジュールは、マルチエージェントシステムで利用可能なエージェントの役割、
能力、接続エンドポイントを一元管理します。

エージェント間で助けを求める際に、このレジストリを参照して最適なエージェントを選択し、
適切なエンドポイントに問い合わせを送信できます。

参考リポジトリ:
- Multi-Agent-Platform: https://github.com/kota-kawa/Multi-Agent-Platform
- IoT-Agent: https://github.com/kota-kawa/IoT-Agent  
- web_agent02 (Browser Agent): https://github.com/kota-kawa/web_agent02
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """エージェントタイプの列挙型"""
    FAQ = "faq"
    BROWSER = "browser"
    IOT = "iot"


@dataclass
class AgentEndpoint:
    """エージェントのエンドポイント情報"""
    url: str
    method: str = "POST"
    description: str = ""


@dataclass
class AgentCapability:
    """エージェントの能力定義"""
    name: str
    description: str
    endpoint: Optional[AgentEndpoint] = None
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class AgentInfo:
    """
    エージェントの詳細情報
    
    Attributes:
        agent_type: エージェントタイプ
        display_name: 表示名（日本語）
        description: エージェントの役割説明
        capabilities: エージェントが提供する機能のリスト
        base_urls: 接続先ベースURL（優先順位順、環境変数から取得可能）
        aliases: エージェントの別名リスト
    """
    agent_type: AgentType
    display_name: str
    description: str
    capabilities: List[AgentCapability]
    base_urls: List[str]
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


# ================================================================
# FAQ エージェント（家庭内エージェント）
# ================================================================
FAQ_AGENT = AgentInfo(
    agent_type=AgentType.FAQ,
    display_name="家庭内エージェント",
    description=(
        "家庭内の出来事や家電に関する専門知識を持つエージェント。\n"
        "FAQデータベース、取扱説明書、家庭内デバイスのナレッジベースに基づいて質問に回答します。\n"
        "IoTデバイスの使い方、トラブルシューティング、メンテナンス方法などの情報を提供できます。"
    ),
    capabilities=[
        AgentCapability(
            name="rag_answer",
            description="ベクトルデータベースから関連情報を検索し、質問に回答",
            endpoint=AgentEndpoint(
                url="/rag_answer",
                method="POST",
                description="質問を受け取り、RAGベースで回答を生成"
            ),
            examples=[
                "エアコンの設定温度は何度がおすすめですか？",
                "洗濯機のエラーコードE01の意味を教えて",
                "冷蔵庫の省エネモードの使い方"
            ]
        ),
        AgentCapability(
            name="agent_rag_answer",
            description="他エージェントからの問い合わせに応答（会話履歴に保存しない）",
            endpoint=AgentEndpoint(
                url="/agent_rag_answer",
                method="POST",
                description="エージェント間通信用のRAG回答エンドポイント"
            ),
            examples=[
                "照明の色温度設定について教えて（エージェント経由）"
            ]
        ),
        AgentCapability(
            name="analyze_conversation",
            description="外部エージェントの会話履歴を分析し、支援が必要か判断",
            endpoint=AgentEndpoint(
                url="/analyze_conversation",
                method="POST",
                description="会話履歴を受け取り、VDBの知識で解決できる問題があれば支援メッセージを返す"
            ),
            examples=[
                "会話履歴から家電の問題を検出して支援"
            ]
        ),
        AgentCapability(
            name="conversation_history",
            description="会話履歴の取得",
            endpoint=AgentEndpoint(
                url="/conversation_history",
                method="GET",
                description="現在の会話履歴を取得"
            )
        ),
        AgentCapability(
            name="conversation_summary",
            description="会話の要約を取得",
            endpoint=AgentEndpoint(
                url="/conversation_summary",
                method="GET",
                description="会話内容の要約を生成"
            )
        ),
        AgentCapability(
            name="reset_history",
            description="会話履歴のリセット",
            endpoint=AgentEndpoint(
                url="/reset_history",
                method="POST",
                description="会話履歴をクリア"
            )
        )
    ],
    base_urls=[
        os.getenv("FAQ_GEMINI_API_BASE", "http://localhost:5000"),
        "http://faq_gemini:5000",
        "http://localhost:5000"
    ],
    aliases=["faq_gemini", "gemini", "knowledge", "knowledge_base", "docs", "home"]
)


# ================================================================
# IoT エージェント
# ================================================================
IOT_AGENT = AgentInfo(
    agent_type=AgentType.IOT,
    display_name="IoT エージェント",
    description=(
        "IoTデバイスの状態確認と操作を行うエージェント。\n"
        "Raspberry Piなどのエッジデバイスと連携し、センサーデータの取得、\n"
        "アクチュエーターの制御、デバイスの管理を担当します。\n"
        "LLMを使用してユーザーの自然言語による指示をデバイスコマンドに変換し、実行します。"
    ),
    capabilities=[
        AgentCapability(
            name="chat",
            description="自然言語でIoTデバイスと対話",
            endpoint=AgentEndpoint(
                url="/api/chat",
                method="POST",
                description="LLMを使用してユーザー指示をデバイスコマンドに変換して実行"
            ),
            examples=[
                "リビングの照明をつけて",
                "温度センサーの値を教えて",
                "エアコンを25度に設定して"
            ]
        ),
        AgentCapability(
            name="agents_respond",
            description="他エージェントからのスポット問い合わせに応答",
            endpoint=AgentEndpoint(
                url="/api/agents/respond",
                method="POST",
                description="ピアエージェントからのIoT環境に関する質問に回答"
            ),
            examples=[
                "現在の室温は？（エージェント経由）",
                "登録されているデバイスの状態（エージェント経由）"
            ]
        ),
        AgentCapability(
            name="conversations_review",
            description="会話履歴を評価してIoT操作が必要か判断",
            endpoint=AgentEndpoint(
                url="/api/conversations/review",
                method="POST",
                description="他エージェントの会話ログからIoT操作の必要性を評価し、必要なら実行"
            ),
            examples=[
                "会話から「部屋が暑い」を検出して冷房を提案"
            ]
        ),
        AgentCapability(
            name="devices_register",
            description="新しいデバイスの登録",
            endpoint=AgentEndpoint(
                url="/api/devices/register",
                method="POST",
                description="IoTデバイスの手動登録とメタ情報の保存"
            )
        ),
        AgentCapability(
            name="devices_list",
            description="登録済みデバイス一覧の取得",
            endpoint=AgentEndpoint(
                url="/api/devices",
                method="GET",
                description="全ての登録済みデバイスの情報を取得"
            )
        ),
        AgentCapability(
            name="device_jobs",
            description="デバイスジョブの作成と管理",
            endpoint=AgentEndpoint(
                url="/api/devices/<device_id>/jobs",
                method="POST",
                description="特定デバイスへのジョブ投入と実行結果の取得"
            )
        )
    ],
    base_urls=[
        os.getenv("IOT_AGENT_API_BASE", "https://iot-agent.project-kk.com"),
        "https://iot-agent.project-kk.com",
        "http://localhost:5006"
    ],
    aliases=["iot_agent", "device", "devices", "sensor", "actuator"]
)


# ================================================================
# ブラウザエージェント
# ================================================================
BROWSER_AGENT = AgentInfo(
    agent_type=AgentType.BROWSER,
    display_name="ブラウザエージェント",
    description=(
        "Webブラウザを自動操作するエージェント。\n"
        "ウェブページの閲覧、情報検索、フォーム入力、ボタンクリックなどの\n"
        "ブラウザ操作を自然言語の指示で実行します。\n"
        "browser-useライブラリとPlaywrightを使用し、Chrome/Chromiumを制御します。"
    ),
    capabilities=[
        AgentCapability(
            name="chat",
            description="自然言語でブラウザ操作を実行",
            endpoint=AgentEndpoint(
                url="/api/chat",
                method="POST",
                description="ユーザーの指示を受け取り、ブラウザ操作を実行して結果を返す"
            ),
            examples=[
                "Googleで「今日の天気」を検索して",
                "Amazonで商品価格を調べて",
                "ニュースサイトの最新記事を要約して"
            ]
        ),
        AgentCapability(
            name="check_conversation_history",
            description="会話履歴をチェックしてブラウザ操作の必要性を判断",
            endpoint=AgentEndpoint(
                url="/api/check-conversation-history",
                method="POST",
                description="他エージェントの会話履歴からWeb検索や操作が必要か評価"
            ),
            examples=[
                "会話から製品情報の検索が必要と判断"
            ]
        ),
        AgentCapability(
            name="agent_relay",
            description="他エージェントからのリレー要求を処理",
            endpoint=AgentEndpoint(
                url="/api/agent-relay",
                method="POST",
                description="エージェント間通信用のリレーエンドポイント"
            )
        ),
        AgentCapability(
            name="history",
            description="ブラウザエージェントの操作履歴を取得",
            endpoint=AgentEndpoint(
                url="/api/history",
                method="GET",
                description="過去の操作メッセージと結果を取得"
            )
        ),
        AgentCapability(
            name="stream",
            description="リアルタイムでブラウザ操作の進捗を取得",
            endpoint=AgentEndpoint(
                url="/api/stream",
                method="GET",
                description="Server-Sent Events (SSE) で操作の進捗をストリーミング"
            )
        ),
        AgentCapability(
            name="reset",
            description="ブラウザセッションのリセット",
            endpoint=AgentEndpoint(
                url="/api/reset",
                method="POST",
                description="ブラウザ状態と履歴をクリア"
            )
        ),
        AgentCapability(
            name="pause",
            description="ブラウザ操作の一時停止",
            endpoint=AgentEndpoint(
                url="/api/pause",
                method="POST",
                description="現在実行中のブラウザ操作を一時停止"
            )
        ),
        AgentCapability(
            name="resume",
            description="ブラウザ操作の再開",
            endpoint=AgentEndpoint(
                url="/api/resume",
                method="POST",
                description="一時停止したブラウザ操作を再開"
            )
        )
    ],
    base_urls=[
        os.getenv("BROWSER_AGENT_API_BASE", "http://localhost:5005"),
        "http://browser-agent:5005",
        "http://browser_agent:5005",
        "http://localhost:5005"
    ],
    aliases=["browser_agent", "web", "web_agent", "navigator", "playwright"]
)


# ================================================================
# エージェントレジストリ
# ================================================================
class AgentRegistry:
    """
    エージェントレジストリ
    
    全てのエージェント情報を管理し、エージェント間通信を支援します。
    """
    
    def __init__(self):
        """レジストリを初期化"""
        self._agents: Dict[AgentType, AgentInfo] = {
            AgentType.FAQ: FAQ_AGENT,
            AgentType.IOT: IOT_AGENT,
            AgentType.BROWSER: BROWSER_AGENT
        }
        
        # 別名からエージェントタイプへのマッピングを構築
        self._alias_map: Dict[str, AgentType] = {}
        for agent_type, agent_info in self._agents.items():
            # エージェントタイプ自体も追加
            self._alias_map[agent_type.value] = agent_type
            # 別名を追加
            for alias in agent_info.aliases:
                self._alias_map[alias.lower()] = agent_type
    
    def get_agent(self, identifier: str) -> Optional[AgentInfo]:
        """
        エージェント識別子（タイプまたは別名）からエージェント情報を取得
        
        Args:
            identifier: エージェントタイプまたは別名
            
        Returns:
            エージェント情報、見つからない場合はNone
        """
        agent_type = self._alias_map.get(identifier.lower())
        if agent_type:
            return self._agents.get(agent_type)
        return None
    
    def get_all_agents(self) -> List[AgentInfo]:
        """全てのエージェント情報を取得"""
        return list(self._agents.values())
    
    def get_agent_by_capability(self, capability_name: str) -> List[AgentInfo]:
        """
        特定の能力を持つエージェントを検索
        
        Args:
            capability_name: 能力名
            
        Returns:
            該当するエージェントのリスト
        """
        result = []
        for agent_info in self._agents.values():
            for capability in agent_info.capabilities:
                if capability.name == capability_name:
                    result.append(agent_info)
                    break
        return result
    
    def get_optimal_agent(self, task_description: str) -> Optional[AgentInfo]:
        """
        タスク説明から最適なエージェントを選択
        
        Args:
            task_description: タスクの説明（自然言語）
            
        Returns:
            最適と思われるエージェント情報
            
        Note:
            現在は簡易的なキーワードマッチング。
            本番環境ではLLMを使用した選択も可能。
        """
        task_lower = task_description.lower()
        
        # ブラウザ関連のキーワード（優先度高）
        browser_keywords = ["検索", "ウェブ", "web", "ブラウザ", "サイト", 
                           "ページ", "google", "amazon", "情報を調べ", "yahoo"]
        if any(keyword in task_lower for keyword in browser_keywords):
            return self._agents[AgentType.BROWSER]
        
        # FAQ/知識ベース関連のキーワード（優先度中）
        faq_keywords = ["使い方", "方法", "教えて", "どうやって", "トラブル",
                       "エラー", "説明書", "マニュアル", "について", "とは"]
        if any(keyword in task_lower for keyword in faq_keywords):
            return self._agents[AgentType.FAQ]
        
        # IoT関連のキーワード（優先度低、操作を伴う場合）
        iot_action_keywords = ["設定", "つけて", "消して", "オン", "オフ", 
                              "制御", "動かし", "止めて", "on", "off"]
        iot_device_keywords = ["デバイス", "センサー", "温度", "照明",
                              "iot", "device", "sensor", "light", "温度計"]
        
        has_action = any(keyword in task_lower for keyword in iot_action_keywords)
        has_device = any(keyword in task_lower for keyword in iot_device_keywords)
        
        if has_action or has_device:
            return self._agents[AgentType.IOT]
        
        # デフォルトはFAQエージェント
        return self._agents[AgentType.FAQ]
    
    def format_agent_list(self, markdown: bool = False) -> str:
        """
        エージェント一覧を整形して返す
        
        Args:
            markdown: Markdown形式で出力するか
            
        Returns:
            整形されたエージェント一覧
        """
        if markdown:
            lines = ["# 利用可能なエージェント\n"]
            for agent_info in self._agents.values():
                lines.append(f"## {agent_info.display_name} ({agent_info.agent_type.value})")
                lines.append(f"\n{agent_info.description}\n")
                lines.append("### 主な機能:")
                for cap in agent_info.capabilities[:3]:  # 主要な3つを表示
                    lines.append(f"- **{cap.name}**: {cap.description}")
                lines.append("")
        else:
            lines = ["利用可能なエージェント:\n"]
            for agent_info in self._agents.values():
                lines.append(f"- {agent_info.display_name} ({agent_info.agent_type.value})")
                lines.append(f"  {agent_info.description}")
                lines.append(f"  接続先: {agent_info.base_urls[0]}")
                lines.append("")
        
        return "\n".join(lines)
    
    def get_agent_endpoints_summary(self, agent_type: AgentType) -> Dict[str, Any]:
        """
        エージェントのエンドポイント情報をまとめて返す
        
        Args:
            agent_type: エージェントタイプ
            
        Returns:
            エンドポイント情報の辞書
        """
        agent_info = self._agents.get(agent_type)
        if not agent_info:
            return {}
        
        return {
            "agent_type": agent_type.value,
            "display_name": agent_info.display_name,
            "description": agent_info.description,
            "base_urls": agent_info.base_urls,
            "endpoints": [
                {
                    "capability": cap.name,
                    "description": cap.description,
                    "url": cap.endpoint.url if cap.endpoint else None,
                    "method": cap.endpoint.method if cap.endpoint else None
                }
                for cap in agent_info.capabilities
            ]
        }


# グローバルレジストリインスタンス
registry = AgentRegistry()


# ================================================================
# ヘルパー関数
# ================================================================

def get_agent_info(identifier: str) -> Optional[AgentInfo]:
    """エージェント情報を取得するヘルパー関数"""
    return registry.get_agent(identifier)


def get_optimal_agent_for_task(task: str) -> Optional[AgentInfo]:
    """タスクに最適なエージェントを選択するヘルパー関数"""
    return registry.get_optimal_agent(task)


def list_all_agents() -> str:
    """全エージェントの一覧を文字列で返す"""
    return registry.format_agent_list()


def list_all_agents_markdown() -> str:
    """全エージェントの一覧をMarkdown形式で返す"""
    return registry.format_agent_list(markdown=True)


if __name__ == "__main__":
    # テスト実行
    print("=== エージェントレジストリのテスト ===\n")
    print(list_all_agents())
    print("\n=== FAQ エージェントの詳細 ===")
    faq = get_agent_info("faq")
    if faq:
        print(f"名前: {faq.display_name}")
        print(f"説明: {faq.description}")
        print(f"接続先: {faq.base_urls[0]}")
        print(f"能力数: {len(faq.capabilities)}")
