#!/usr/bin/env python3
"""
エージェントレジストリとエンドポイントのテスト

このスクリプトはagent_registry.pyとapp.pyの新機能をテストします。
"""

import sys
import os

# テスト用のモックを追加
sys.path.insert(0, os.path.dirname(__file__))

def test_agent_registry():
    """agent_registry.pyの基本機能をテスト"""
    print("=== エージェントレジストリのテスト ===\n")
    
    from agent_registry import (
        get_agent_info,
        get_optimal_agent_for_task,
        list_all_agents,
        AgentType,
        registry
    )
    
    # テスト1: エージェント情報の取得
    print("テスト1: エージェント情報の取得")
    faq_agent = get_agent_info("faq")
    assert faq_agent is not None, "FAQエージェントが見つかりません"
    assert faq_agent.display_name == "家庭内エージェント"
    print(f"✓ FAQエージェント取得成功: {faq_agent.display_name}")
    
    # テスト2: 別名でのエージェント取得
    print("\nテスト2: 別名でのエージェント取得")
    iot_agent = get_agent_info("device")  # "device"は"iot"の別名
    assert iot_agent is not None, "IoTエージェントが別名で取得できません"
    assert iot_agent.agent_type == AgentType.IOT
    print(f"✓ IoTエージェント別名取得成功: {iot_agent.display_name}")
    
    browser_agent = get_agent_info("web")  # "web"は"browser"の別名
    assert browser_agent is not None, "ブラウザエージェントが別名で取得できません"
    assert browser_agent.agent_type == AgentType.BROWSER
    print(f"✓ ブラウザエージェント別名取得成功: {browser_agent.display_name}")
    
    # テスト3: タスクからエージェントを選択
    print("\nテスト3: タスクからエージェントを選択")
    test_cases = [
        ("部屋の温度を25度に設定して", AgentType.IOT),
        ("Googleで天気を検索して", AgentType.BROWSER),
        ("エアコンの使い方を教えて", AgentType.FAQ),
    ]
    
    for task, expected_type in test_cases:
        optimal = get_optimal_agent_for_task(task)
        assert optimal is not None, f"タスク '{task}' に対するエージェントが見つかりません"
        assert optimal.agent_type == expected_type, \
            f"タスク '{task}' の推奨エージェントが誤っています: {optimal.agent_type} != {expected_type}"
        print(f"✓ タスク '{task[:20]}...' → {optimal.display_name}")
    
    # テスト4: 全エージェント一覧
    print("\nテスト4: 全エージェント一覧の取得")
    all_agents = registry.get_all_agents()
    assert len(all_agents) == 3, f"エージェント数が不正です: {len(all_agents)}"
    print(f"✓ 全エージェント数: {len(all_agents)}")
    
    # テスト5: 能力によるエージェント検索
    print("\nテスト5: 能力によるエージェント検索")
    chat_agents = registry.get_agent_by_capability("chat")
    assert len(chat_agents) == 2, f"chatを持つエージェント数が不正です: {len(chat_agents)}"
    print(f"✓ 'chat'能力を持つエージェント: {[a.display_name for a in chat_agents]}")
    
    # テスト6: エンドポイント情報の取得
    print("\nテスト6: エンドポイント情報の取得")
    endpoints = registry.get_agent_endpoints_summary(AgentType.FAQ)
    assert "endpoints" in endpoints
    assert len(endpoints["endpoints"]) > 0
    print(f"✓ FAQエージェントのエンドポイント数: {len(endpoints['endpoints'])}")
    
    print("\n全テスト成功! ✓")
    return True


def test_agent_info_display():
    """エージェント情報の表示テスト"""
    print("\n\n=== エージェント情報の表示 ===\n")
    
    from agent_registry import registry, AgentType
    
    for agent_type in [AgentType.FAQ, AgentType.IOT, AgentType.BROWSER]:
        agent_info = registry.get_agent_endpoints_summary(agent_type)
        print(f"\n【{agent_info['display_name']}】")
        print(f"タイプ: {agent_info['agent_type']}")
        print(f"説明: {agent_info['description'][:100]}...")
        print(f"接続先: {agent_info['base_urls'][0]}")
        print(f"主な機能:")
        for endpoint in agent_info['endpoints'][:3]:  # 主要な3つを表示
            print(f"  - {endpoint['capability']}: {endpoint['description'][:60]}...")
    
    return True


def test_module_imports():
    """モジュールのインポートテスト"""
    print("\n\n=== モジュールインポートのテスト ===\n")
    
    try:
        # agent_registryのインポート
        from agent_registry import (
            registry,
            get_agent_info,
            get_optimal_agent_for_task,
            list_all_agents,
            AgentType,
            AgentInfo,
            AgentCapability,
            AgentEndpoint
        )
        print("✓ agent_registry.py のインポート成功")
        
        # dataclassの確認
        assert hasattr(AgentInfo, '__dataclass_fields__')
        print("✓ AgentInfo dataclass 確認")
        
        assert hasattr(AgentCapability, '__dataclass_fields__')
        print("✓ AgentCapability dataclass 確認")
        
        assert hasattr(AgentEndpoint, '__dataclass_fields__')
        print("✓ AgentEndpoint dataclass 確認")
        
        # Enumの確認
        assert hasattr(AgentType, 'FAQ')
        assert hasattr(AgentType, 'IOT')
        assert hasattr(AgentType, 'BROWSER')
        print("✓ AgentType Enum 確認")
        
        print("\n全インポート成功! ✓")
        return True
        
    except Exception as e:
        print(f"✗ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全テストを実行"""
    print("=" * 60)
    print("FAQ_Gemini エージェントレジストリテスト")
    print("=" * 60)
    
    try:
        # テスト実行
        success = True
        success = test_module_imports() and success
        success = test_agent_registry() and success
        success = test_agent_info_display() and success
        
        print("\n" + "=" * 60)
        if success:
            print("全てのテストが成功しました! ✓")
            print("=" * 60)
            return 0
        else:
            print("一部のテストが失敗しました ✗")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\n✗ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
