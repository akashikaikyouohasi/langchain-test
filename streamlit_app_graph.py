import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langfuse import Langfuse

# agents_graph.pyからエージェントをインポート
from agents_graph import agent_graph

# 環境変数のロード
from dotenv import load_dotenv
load_dotenv()

def init_session_state():
    """セッション状態を初期化する"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'waiting_for_approval' not in st.session_state:
        st.session_state.waiting_for_approval = False
    if 'final_result' not in st.session_state:
        st.session_state.final_result = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "trace_id" not in st.session_state:
        st.session_state.trace_id = None

def reset_session():
    """セッション状態をリセットする"""
    st.session_state.messages = []
    st.session_state.waiting_for_approval = False
    st.session_state.final_result = None
    st.session_state.thread_id = None
    st.session_state.trace_id = None

# セッション状態の初期化を実行
init_session_state()

def run_agent(input_data):
    """エージェントを実行し、結果を処理する"""
    # trace_idがまだなければ、thread_idから生成
    if not st.session_state.trace_id:
        st.session_state.trace_id = Langfuse.create_trace_id(seed=st.session_state.thread_id)
        print(f"[Streamlit] 新しいtrace_idを生成: {st.session_state.trace_id}")
    
    # LangfuseのCallbackHandlerを初期化（引数なし）
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
    
    # LangGraphの設定
    # metadataでsession_id, user_id, tagsを設定してinterruptの前後をつなげる
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": st.session_state.thread_id,  # interruptの前後で同じsession_id
            "langfuse_tags": ["graph-api", "with-interrupt"]
        }
    }
    
    # input_dataがCommandの場合はそのまま、それ以外はstateとして渡す
    if isinstance(input_data, Command):
        stream_input = input_data
    else:
        # input_dataは既に[HumanMessage(...)]のリストなので、messagesキーで渡す
        # trace_idもstateに含める
        stream_input = {
            "messages": input_data,  # これは[HumanMessage(...)]のリスト
            "trace_id": st.session_state.trace_id
        }
    
    print(f"[Streamlit] trace_id: {st.session_state.trace_id}")
    print(f"[Streamlit] session_id: {st.session_state.thread_id}")
    
    # 結果を処理
    with st.spinner("処理中...", show_time=True):
        for event in agent_graph.stream(stream_input, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                print(f"[Streamlit] イベント受信: {node_name}")
                
                # interruptの場合
                if node_name == "__interrupt__":
                    st.session_state.tool_info = node_output[0].value
                    st.session_state.waiting_for_approval = True
                
                # agentノードからの出力
                elif node_name == "agent":
                    # 最後のメッセージを取得
                    if "messages" in node_output and node_output["messages"]:
                        last_msg = node_output["messages"][-1]
                        # ツール呼び出しがない場合は最終結果として扱う
                        if hasattr(last_msg, 'content') and not getattr(last_msg, 'tool_calls', None):
                            st.session_state.final_result = last_msg.content
        
        st.rerun()

def feedback():
    """フィードバックを取得し、エージェントに通知する関数"""       
    approve_column, deny_column = st.columns(2)

    feedback_result = None
    with approve_column:
        if st.button("APPROVE", width="stretch"):
            st.session_state.waiting_for_approval = False
            feedback_result = "APPROVE"
    with deny_column:
        if st.button("DENY", width="stretch"):
            st.session_state.waiting_for_approval = False
            feedback_result = "DENY"
                
    return feedback_result

def app():
    # タイトルの設定
    st.title("Webリサーチエージェント (Graph API版)")

    # メッセージ表示エリア
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
            
    # ツール承認の確認
    if st.session_state.waiting_for_approval \
       and st.session_state.tool_info:
        st.info(st.session_state.tool_info["args"])
        if st.session_state.tool_info["name"] == "write_file":
            with st.container(height=400):
                st.html(st.session_state.tool_info["html"], width="stretch")
        feedback_result = feedback()
        if feedback_result:
            st.chat_message("user").write(feedback_result)
            st.session_state.messages.append({"role": "user", "content": feedback_result})
            # interruptの再開コマンドを送信
            run_agent(Command(resume=feedback_result))
            st.rerun()

    # 最終結果の表示
    if st.session_state.final_result \
       and not st.session_state.waiting_for_approval:
        st.subheader("最終結果")
        st.success(st.session_state.final_result)

    # ユーザー入力エリア
    if not st.session_state.waiting_for_approval:
        user_input = st.chat_input("メッセージを入力してください")
        if user_input:
            reset_session()
            # スレッドIDを設定
            st.session_state.thread_id = str(uuid.uuid4())
            # ユーザーメッセージを追加
            st.chat_message("user").write(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # エージェントを実行
            message = HumanMessage(content=user_input, id=st.session_state.thread_id)
            # run_agentにはメッセージのリストを渡す
            if run_agent([message]):
                st.rerun()
    else:
        st.info("ツールの承認待ちです。上記のボタンで応答してください。")

# メインの実行
if __name__ == "__main__":
    app()
