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
    # 実際のWeb検索APIを使用する場合はここを実装
    return f"'{query}'についての検索結果: これはサンプルの検索結果です。"


@tool
def calculator(expression: str) -> str:
    """数式を計算します。例: '2 + 2' や '10 * 5'"""
    try:
        # 安全性のため、evalの代わりに制限された計算を行う
        result = eval(expression, {"__builtins__": {}}, {})
        return f"計算結果: {result}"
    except Exception as e:
        return f"エラー: {str(e)}"


@tool
def get_current_info(topic: str) -> str:
    """特定のトピックについての現在の情報を取得します。"""
    return f"'{topic}'についての現在の情報: これはサンプル情報です。"


# ツールのリスト
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
    
    # 入力プロンプトをログ出力
    print("\n" + "="*50)
    print("🤖 Agent Node - 入力プロンプト")
    print("="*50)
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        print(f"\n[メッセージ {i}] {msg_type}")
        if hasattr(msg, "content") and msg.content:
            print(f"Content: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Tool Calls: {msg.tool_calls}")
        if isinstance(msg, ToolMessage):
            print(f"Tool Call ID: {msg.tool_call_id}")
    
    # LLMを呼び出し
    response = llm_with_tools.invoke(messages)
    
    # 出力プロンプトをログ出力
    print("\n" + "="*50)
    print("🤖 Agent Node - 出力プロンプト")
    print("="*50)
    print(f"Response Type: {type(response).__name__}")
    if hasattr(response, "content") and response.content:
        print(f"Content: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nTool Calls ({len(response.tool_calls)}件):")
        for tc in response.tool_calls:
            print(f"  - ツール: {tc['name']}")
            print(f"    引数: {tc['args']}")
            print(f"    ID: {tc['id']}")
    print("="*50 + "\n")
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "human_review", "finalize"]:
    """次に進むべきノードを決定"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # ツール呼び出しがある場合
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 重要な操作（例: calculator）の場合は人間のレビューを要求
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        if "calculator" in tool_names:
            return "human_review"
        return "tools"
    
    # ツール呼び出しがない場合は最終化
    return "finalize"


def human_review_node(state: AgentState) -> dict:
    """
    人間のレビューを待つノード（interrupt使用）
    
    このノードはinterrupt()を呼び出してグラフを一時停止します。
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
        # 承認された場合、何も返さない（次のノードへ進む）
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
    
    # Pydanticモデルで構造化された出力を生成
    structured_llm = llm.with_structured_output(FinalAnswer)
    
    # 会話履歴から最終回答を生成
    messages = state["messages"]
    final_prompt = HumanMessage(
        content="これまでの会話内容を基に、構造化された最終回答を生成してください。"
    )
    
    # 入力プロンプトをログ出力
    print("\n" + "="*50)
    print("📝 Finalize Node - 入力プロンプト")
    print("="*50)
    print(f"メッセージ履歴: {len(messages)}件")
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        print(f"\n[メッセージ {i}] {msg_type}")
        if hasattr(msg, "content") and msg.content:
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"Content: {content_preview}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Tool Calls: {len(msg.tool_calls)}件")
        if isinstance(msg, ToolMessage):
            print(f"Tool Call ID: {msg.tool_call_id}")
    
    print(f"\n[追加プロンプト] {type(final_prompt).__name__}")
    print(f"Content: {final_prompt.content}")
    print("="*50)
    
    # LLMを呼び出し
    final_answer = structured_llm.invoke(list(messages) + [final_prompt])
    
    # 出力プロンプト（構造化された結果）をログ出力
    print("\n" + "="*50)
    print("📝 Finalize Node - 出力プロンプト（構造化された結果）")
    print("="*50)
    print(f"型: {type(final_answer).__name__}")
    print(f"\n要約: {final_answer.summary}")
    print(f"\n発見事項 ({len(final_answer.findings)}件):")
    for i, finding in enumerate(final_answer.findings, 1):
        print(f"  {i}. {finding}")
    print(f"\n計算結果: {final_answer.calculations}")
    print(f"信頼度: {final_answer.confidence}")
    print(f"情報源: {final_answer.sources}")
    print("="*50 + "\n")
    
    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content="最終回答を生成しました。")]
    }


# ========== グラフの構築 ==========

def create_agent_graph():
    """エージェントグラフを作成"""
    # ツールノードを作成
    tool_node = ToolNode(tools)
    
    # グラフを初期化
    workflow = StateGraph(AgentState)
    
    # ノードを追加
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize", finalize_node)
    
    # エントリーポイントを設定
    workflow.set_entry_point("agent")
    
    # エッジを追加
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "human_review": "human_review",
            "finalize": "finalize"
        }
    )
    
    # human_reviewの後はtoolsへ（承認時）またはagentへ（拒否時）
    # human_review_nodeの返り値で判断
    def after_human_review(state: AgentState) -> Literal["tools", "agent"]:
        messages = state["messages"]
        if messages:
            last_msg = messages[-1]
            # ToolMessageがある = 拒否された
            if isinstance(last_msg, ToolMessage):
                return "agent"
        # それ以外は承認 = toolsへ
        return "tools"
    
    workflow.add_conditional_edges(
        "human_review",
        after_human_review,
        {
            "tools": "tools",
            "agent": "agent"
        }
    )
    
    # ツール実行後はエージェントに戻る
    workflow.add_edge("tools", "agent")
    
    # 最終化後は終了
    workflow.add_edge("finalize", END)
    
    # メモリを追加（会話の状態を保持）
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# ========== メイン実行 ==========

def main():
    """メイン実行関数（interrupt対応）"""
    print("LangGraph エージェント with Human-in-the-Loop (Interrupt版)")
    print("="*50)
    
    # グラフを作成
    app = create_agent_graph()
    
    # 初期メッセージ
    initial_message = input("\nタスクを入力してください > ")
    
    # 設定
    config = {"configurable": {"thread_id": "1"}}
    
    # 初期状態
    initial_state = {
        "messages": [HumanMessage(content=initial_message)],
        "final_answer": None
    }
    
    print("\n処理を開始します...\n")
    
    # グラフを実行（interrupt対応ループ）
    current_state = initial_state
    
    while True:
        # グラフを実行
        for event in app.stream(current_state, config, stream_mode="values"):
            # デバッグ出力
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                msg_type = type(last_msg).__name__
                if hasattr(last_msg, "content") and last_msg.content:
                    print(f"💬 {msg_type}: {last_msg.content[:100]}...")
                elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"🔧 {msg_type}: ツール呼び出し {len(last_msg.tool_calls)}件")
            
            # 最終結果をチェック
            if event.get("final_answer"):
                print("\n" + "="*50)
                print("✅ 最終的な構造化された出力")
                print("="*50)
                final_answer = event["final_answer"]
                print(f"\n要約: {final_answer.summary}")
                print(f"\n発見事項:")
                for i, finding in enumerate(final_answer.findings, 1):
                    print(f"  {i}. {finding}")
                print(f"\n計算結果: {final_answer.calculations}")
                print(f"信頼度: {final_answer.confidence}")
                print(f"情報源: {final_answer.sources}")
                return
        
        # interruptが発生したかチェック
        snapshot = app.get_state(config)
        
        if not snapshot.next:
            # 次のノードがない = 完了
            break
        
        # interruptが発生している場合
        if snapshot.tasks:
            task = snapshot.tasks[0]
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                
                print("\n" + "="*50)
                print("🔍 Human Review Required")
                print("="*50)
                
                # ツール情報を表示
                for tool_call in interrupt_value.get("tool_calls", []):
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
                else:
                    # 拒否してフィードバック
                    feedback = input("フィードバックを入力してください > ")
                    app.update_state(
                        config,
                        {"approved": False, "feedback": feedback},
                        as_node="human_review"
                    )
                
                # 次のループで続行
                current_state = None
                continue
        
        # interruptがない場合は終了
        break


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # 環境変数を読み込み
    load_dotenv()
    
    if not os.getenv("AWS_BEDROCK_MODEL"):
        print("⚠️  AWS Bedrockの設定を確認してください。")
        print(".envファイルでAWS_BEDROCK_MODELを設定するか、")
        print("AWS CLIで認証情報を設定してください。")
        print("\nAWS CLIの設定: aws configure")
        print("\n💡 interrupt版の実装:")
        print("   - グラフが一時停止し、外部から再開可能")
        print("   - Web UI、API、Slackなどと統合可能")
        print("   - 状態が永続化され、後から再開可能")
    else:
        main()
