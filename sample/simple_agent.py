"""
簡易版: Human-in-the-Loop付きエージェント

より簡単に理解できるシンプルなバージョンです。
"""

import os
from typing import Annotated, Sequence, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ========== シンプルなツール ==========

@tool
def add_numbers(a: float, b: float) -> float:
    """2つの数を足し算します"""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """2つの数を掛け算します"""
    return a * b


tools = [add_numbers, multiply_numbers]


# ========== 構造化された出力 ==========

class CalculationResult(BaseModel):
    """計算結果の構造化された出力"""
    question: str = Field(description="元の質問")
    steps: list[str] = Field(description="実行したステップ")
    final_result: str = Field(description="最終結果")


# ========== 状態 ==========

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    approved: bool


# ========== ノード ==========

def agent_node(state: State):
    """エージェント: 次のアクションを決定"""
    llm = ChatBedrock(
        model_id=os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        model_kwargs={"temperature": 0}
    )
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_approval_node(state: State):
    """人間の承認を求める"""
    last_message = state["messages"][-1]
    
    print("\n" + "="*50)
    print("🤔 承認が必要です")
    print("="*50)
    
    if hasattr(last_message, "tool_calls"):
        for tc in last_message.tool_calls:
            print(f"ツール: {tc['name']}")
            print(f"引数: {tc['args']}")
    
    approval = input("\n承認しますか？ (y/n) > ").strip().lower()
    
    return {"approved": approval in ["y", "yes"]}


def finalize_node(state: State):
    """最終的な構造化出力を生成"""
    llm = ChatBedrock(
        model_id=os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        model_kwargs={"temperature": 0}
    )
    structured_llm = llm.with_structured_output(CalculationResult)
    
    result = structured_llm.invoke([
        HumanMessage(content="会話履歴から計算結果をまとめてください"),
        *state["messages"]
    ])
    
    print("\n" + "="*50)
    print("✅ 最終結果（構造化）")
    print("="*50)
    print(f"質問: {result.question}")
    print(f"\nステップ:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")
    print(f"\n最終結果: {result.final_result}")
    
    return {"messages": [AIMessage(content="完了")]}


# ========== ルーティング ==========

def should_continue(state: State):
    """次のノードを決定"""
    last_message = state["messages"][-1]
    
    # ツール呼び出しがあるか確認
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "human_approval"
    
    return "finalize"


def after_approval(state: State):
    """承認後のルーティング"""
    if state.get("approved", False):
        return "tools"
    else:
        return "agent"


# ========== グラフ構築 ==========

def create_simple_graph():
    """シンプルなグラフを作成"""
    workflow = StateGraph(State)
    
    # ノード追加
    workflow.add_node("agent", agent_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("finalize", finalize_node)
    
    # フロー設定
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "human_approval": "human_approval",
            "finalize": "finalize"
        }
    )
    
    workflow.add_conditional_edges(
        "human_approval",
        after_approval,
        {
            "tools": "tools",
            "agent": "agent"
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# ========== メイン ==========

def main():
    print("シンプルな Human-in-the-Loop エージェント")
    print("="*50)
    
    app = create_simple_graph()
    
    # 例: 10 + 5 を計算させる
    question = "10 + 5 を計算してください"
    print(f"\n質問: {question}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "approved": False
    }
    
    for output in app.stream(initial_state):
        pass  # ノードが進むたびに処理


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("AWS_BEDROCK_MODEL"):
        print("⚠️  AWS Bedrockの設定を確認してください。")
        print(".envファイルでAWS_BEDROCK_MODELを設定するか、")
        print("AWS CLIで認証情報を設定してください。")
    else:
        main()
