from typing import Annotated, Literal, TypedDict
from typing_extensions import NotRequired
from botocore.config import Config
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    AIMessage, 
    ToolMessage, 
    HumanMessage
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langfuse import get_client, Langfuse
from langfuse.langchain import CallbackHandler

# Langfuse clientを取得
langfuse = get_client()

# 環境変数のロード
from dotenv import load_dotenv
load_dotenv()

# デバッグログ出力
import logging
logging.basicConfig(level=logging.DEBUG)


# ========== ツールの定義 ==========
# Web検索ツール
web_search = TavilySearch(max_results=2, topic="general")

working_directory = "report"
# ローカルファイルを扱うツール
file_toolkit = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["write_file"]  # 書き込みツールのみ有効化
)
write_file = file_toolkit.get_tools()[0]
logging.debug("ファイルツールキット: %s", file_toolkit.get_tools()[0])

# 使用するツールのリスト
tools = [web_search, write_file]
tools_by_name = {tool.name: tool for tool in tools}

# ========== LLMの初期化 ==========
cfg = Config(
    read_timeout=300,
)
llm_with_tools = init_chat_model(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    model_provider="bedrock_converse",
    config=cfg,
).bind_tools(tools)

# システムプロンプト
system_prompt = """
あなたの責務はユーザからのリクエストを調査し、調査結果をファイル出力することです。
- ユーザーのリクエスト調査にWeb検索が必要であれば、Web検索ツールを使ってください。
- 必要な情報が集まったと判断したら検索は終了して下さい。
- 検索は最大2回までとしてください。
- ファイル出力はHTML形式(.html)に変換して保存してください。
  * Web検索が拒否された場合、Web検索を中止してレポート作成してください。
  * レポート保存を拒否された場合、レポート作成を中止し、内容をユーザーに直接伝えて下さい。
"""


# ========== Graph ノードの定義 ==========
# カスタムStateを定義（trace_idを追加）
class AgentState(MessagesState):
    trace_id: NotRequired[str]


def agent_node(state: AgentState) -> dict:
    """LLMを呼び出してツール呼び出しを決定するノード"""
    print(f"[Agent Node] メッセージ数: {len(state['messages'])}")
    print(f"[Agent Node] trace_id: {state.get('trace_id')}")
    
    # trace_idを取得（初回はNoneの可能性あり）
    trace_id = state.get("trace_id")
    
    # システムプロンプトを先頭に追加
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # CallbackHandlerを初期化
    langfuse_handler = CallbackHandler()
    
    # RunnableConfigを作成
    config = RunnableConfig(callbacks=[langfuse_handler])
    if trace_id:
        config["metadata"] = { 
            "langfuse_session_id": trace_id,
            "langfuse_tags": ["random-tag-1", "random-tag-2"]
        }
    
    # LLM呼び出し
    response = llm_with_tools.invoke(messages, config=config)
    
    print(f"[Agent Node] Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")
    
    # メッセージをstateに追加
    return {"messages": [response]}


def human_approval_node(state: AgentState) -> dict:
    """ツール実行前に人間の承認を求め、ツールを実行するノード"""
    last_message = state["messages"][-1]
    
    print(f"[Human Approval Node] trace_id: {state.get('trace_id')}")
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}
    
    tool_messages = _execute_tools_with_approval(last_message.tool_calls)
    
    # すべてのツール結果を返す
    return {"messages": tool_messages}


def _execute_tools_with_approval(tool_calls):
    """ツールの承認と実行を行う内部関数"""
    tool_messages = []
    
    # 各ツール呼び出しに対して承認を求める
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_data = {"name": tool_name}
        
        # ツールごとに表示用データを作成
        if tool_name == web_search.name:
            args = f'* ツール名\n'
            args += f'  * {tool_name}\n'
            args += "* 引数\n"
            for key, value in tool_args.items():
                args += f'  * {key}\n'
                args += f'    * {value}\n'
            tool_data["args"] = args
        elif tool_name == write_file.name:
            args = f'* ツール名\n'
            args += f'  * {tool_name}\n'
            args += f'* 保存ファイル名\n'
            args += f'  * {tool_args["file_path"]}'
            tool_data["args"] = args
            tool_data["html"] = tool_args["text"]
        
        # ユーザーに承認を求める（interrupt）
        feedback = interrupt(tool_data)
        
        if feedback == "APPROVE":
            # 承認されたツールを実行
            tool = tools_by_name[tool_name]
            observation = tool.invoke(tool_args)
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"]
                )
            )
        else:
            # 拒否されたツールの結果メッセージを作成
            tool_messages.append(
                ToolMessage(
                    content="ツール利用が拒否されたため、処理を終了してください。",
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                )
            )
    
    return tool_messages


def should_continue(state: AgentState) -> Literal["human_approval", "end"]:
    """次のノードを決定するルーティング関数"""
    last_message = state["messages"][-1]
    
    # 最後のメッセージがAIメッセージで、ツール呼び出しがある場合
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "human_approval"
    
    # それ以外の場合は終了
    return "end"


# ========== Graph の構築 ==========
# グラフの作成
workflow = StateGraph(AgentState)

# ノードの追加
workflow.add_node("agent", agent_node)
workflow.add_node("human_approval", human_approval_node)

# エッジの追加
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "human_approval": "human_approval",
        "end": END
    }
)
workflow.add_edge("human_approval", "agent")

# チェックポインタの設定
checkpointer = MemorySaver()

# グラフのコンパイル
agent_graph = workflow.compile(checkpointer=checkpointer)
