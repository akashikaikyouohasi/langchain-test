"""
LangGraphを使用したエージェント実装（interrupt版）
- 複数のツール
- Human in the Loop（interrupt使用）
- 構造化された出力
"""

import operator
import os
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# ========== ツールの定義 ==========

@tool
def search_web(query: str) -> str:
    """Webで情報を検索します。"""
    return f"'{query}'についての検索結果: これはサンプルの検索結果です。"


@tool
def calculator(expression: str) -> str:
    """数式を計算します。例: '2 + 2' や '10 * 5'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"計算結果: {result}"
    except Exception as e:
        return f"エラー: {str(e)}"


@tool
def get_current_info(topic: str) -> str:
    """特定のトピックについての現在の情報を取得します。"""
    return f"'{topic}'についての現在の情報: これはサンプル情報です。"


tools = [search_web, calculator, get_current_info]


# ========== 構造化された出力の定義 ==========

class FinalAnswer(BaseModel):
    """エージェントの最終的な構造化された回答"""
    summary: str = Field(description="タスクの要約")
    findings: list[str] = Field(description="発見した重要な情報のリスト")
    calculations: dict[str, float] = Field(
        default_factory=dict,
        description="実行した計算とその結果"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="回答の信頼度（0.0-1.0）"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="使用した情報源"
    )


# ========== グラフの状態定義 ==========

class AgentState(TypedDict):
    """エージェントの状態"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    final_answer: FinalAnswer | None


# ========== ノードの定義 ==========

def agent_node(state: AgentState) -> dict:
    """エージェントノード: LLMを呼び出してアクションを決定"""
    llm = ChatBedrock(
        model_id=os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        model_kwargs={"temperature": 0}
    )
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "human_review", "finalize"]:
    """次に進むべきノードを決定"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        if "calculator" in tool_names:
            return "human_review"
        return "tools"
    
    return "finalize"


def human_review_node(state: AgentState) -> dict:
    """
    人間のレビューを待つノード（interrupt使用）
    
    このノードはinterruptを呼び出し、グラフを一時停止します。
    外部から承認/拒否の応答を受け取るまで待機します。
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # ツール情報を取得
    tool_calls_info = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_calls_info.append({
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call["id"]
            })
    
    # interrupt()を呼び出してグラフを一時停止
    # 戻り値として承認データを期待
    approval_data = interrupt({
        "type": "human_review",
        "tool_calls": tool_calls_info,
        "message": "ツールの実行には承認が必要です"
    })
    
    # approval_dataの形式:
    # {"approved": True} または {"approved": False, "feedback": "..."}
    
    if approval_data.get("approved"):
        # 承認された場合、何も返さない（toolsノードへ進む）
        return {}
    else:
        # 拒否された場合、フィードバックを追加
        feedback = approval_data.get("feedback", "ユーザーが拒否しました")
        return {
            "messages": [
                ToolMessage(
                    content=f"ユーザーがキャンセルしました。フィードバック: {feedback}",
                    tool_call_id=tool_calls_info[0]["id"]
                )
            ]
        }


def finalize_node(state: AgentState) -> dict:
    """最終的な構造化された出力を生成"""
    llm = ChatBedrock(
        model_id=os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        model_kwargs={"temperature": 0}
    )
    
    structured_llm = llm.with_structured_output(FinalAnswer)
    
    messages = state["messages"]
    final_prompt = HumanMessage(
        content="これまでの会話内容を基に、構造化された最終回答を生成してください。"
    )
    
    final_answer = structured_llm.invoke(list(messages) + [final_prompt])
    
    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content="最終回答を生成しました。")]
    }


# ========== グラフの構築 ==========

def create_agent_graph():
    """エージェントグラフを作成"""
    tool_node = ToolNode(tools)
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "human_review": "human_review",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("human_review", "tools")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)
    
    # メモリを追加（状態を永続化）
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# ========== メイン実行（コンソール版） ==========

def main():
    """メイン実行関数（コンソールインターフェース）"""
    print("LangGraph エージェント with Interrupts")
    print("="*50)
    
    app = create_agent_graph()
    
    initial_message = input("\nタスクを入力してください > ")
    
    config = {"configurable": {"thread_id": "1"}}
    
    initial_state = {
        "messages": [HumanMessage(content=initial_message)],
        "final_answer": None
    }
    
    print("\n処理を開始します...\n")
    
    # グラフを実行
    current_state = initial_state
    
    while True:
        try:
            # グラフをストリーム実行
            result = None
            for event in app.stream(current_state, config, stream_mode="values"):
                result = event
                
                # デバッグ出力
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"💬 {type(last_msg).__name__}: {last_msg.content[:100]}...")
            
            # 完了した場合
            if result and result.get("final_answer"):
                print("\n" + "="*50)
                print("✅ 最終的な構造化された出力")
                print("="*50)
                final_answer = result["final_answer"]
                print(f"\n要約: {final_answer.summary}")
                print(f"\n発見事項:")
                for i, finding in enumerate(final_answer.findings, 1):
                    print(f"  {i}. {finding}")
                print(f"\n計算結果: {final_answer.calculations}")
                print(f"信頼度: {final_answer.confidence}")
                print(f"情報源: {final_answer.sources}")
                break
                
        except Exception as e:
            # interruptによる中断をキャッチ
            if "interrupt" in str(type(e)).lower() or hasattr(e, '__cause__'):
                # 最新の状態を取得
                snapshot = app.get_state(config)
                
                # interrupt情報を取得
                if snapshot.tasks:
                    task = snapshot.tasks[0]
                    interrupt_data = task.interrupts[0].value if task.interrupts else None
                    
                    if interrupt_data:
                        print("\n" + "="*50)
                        print("🔍 Human Review Required")
                        print("="*50)
                        
                        for tool_call in interrupt_data.get("tool_calls", []):
                            print(f"\nツール: {tool_call['name']}")
                            print(f"引数: {tool_call['args']}")
                        
                        print("\n承認しますか？")
                        print("  y/yes: 承認して続行")
                        print("  n/no: 拒否してフィードバックを入力")
                        
                        user_input = input("\n入力 > ").strip().lower()
                        
                        if user_input in ["y", "yes"]:
                            # 承認して再開
                            app.update_state(
                                config,
                                {"approved": True},
                                as_node="human_review"
                            )
                            current_state = None  # 既存の状態から続行
                        else:
                            # 拒否してフィードバック
                            feedback = input("フィードバックを入力してください > ")
                            app.update_state(
                                config,
                                {"approved": False, "feedback": feedback},
                                as_node="human_review"
                            )
                            current_state = None
                    else:
                        break
                else:
                    break
            else:
                print(f"\nエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                break


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("AWS_BEDROCK_MODEL"):
        print("⚠️  AWS Bedrockの設定を確認してください。")
        print(".envファイルでAWS_BEDROCK_MODELを設定するか、")
        print("AWS CLIで認証情報を設定してください。")
    else:
        main()
