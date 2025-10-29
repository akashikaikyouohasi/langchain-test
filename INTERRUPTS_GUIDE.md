# LangGraph Interrupts の使い方

## Interrupts とは？

LangGraphの**interrupt**機能を使うと、グラフの実行を一時停止して、外部からの入力（承認など）を待つことができます。

---

## 主な利点

### 従来の`input()`方式の問題点
```python
# ❌ 問題点
def human_review_node(state):
    user_input = input("承認しますか？ > ")  # ← ここでブロック
    # - コンソールでしか使えない
    # - Web UIやAPIと統合できない
    # - 非同期処理ができない
```

### `interrupt`を使った方式
```python
# ✅ 改善点
def human_review_node(state):
    approval = interrupt({"tool_calls": ...})  # ← ここで一時停止
    # - グラフが中断される
    # - 外部システムから状態を更新して再開できる
    # - Web UI、Slack、APIなど任意のインターフェースで承認可能
```

---

## 実装方法

### 基本的な使い方

```python
from langgraph.types import interrupt

def human_review_node(state: AgentState) -> dict:
    """承認が必要なノード"""
    
    # 1. interrupt()を呼び出してグラフを停止
    approval_data = interrupt({
        "type": "human_approval",
        "message": "ツールの実行を承認しますか？",
        "tool_info": {...}
    })
    
    # 2. approval_dataには外部から渡されたデータが入る
    if approval_data.get("approved"):
        return {}  # 承認 → 次のノードへ
    else:
        return {"feedback": approval_data.get("feedback")}  # 拒否
```

---

## グラフ実行の流れ

### 1. 初回実行

```python
app = create_agent_graph()
config = {"configurable": {"thread_id": "1"}}

# グラフを実行
for event in app.stream(initial_state, config):
    print(event)
    # interrupt()に到達すると例外が発生して停止
```

### 2. 中断状態の確認

```python
# 現在の状態を取得
snapshot = app.get_state(config)

# 中断情報を確認
if snapshot.next:  # 次のノードがある = 中断中
    print(f"中断されたノード: {snapshot.next}")
    
    # interrupt()に渡されたデータ
    if snapshot.tasks:
        task = snapshot.tasks[0]
        if task.interrupts:
            interrupt_value = task.interrupts[0].value
            print(f"Interrupt data: {interrupt_value}")
```

### 3. 承認して再開

```python
# 状態を更新して再開
app.update_state(
    config,
    {"approved": True},  # ← interrupt()に渡される値
    as_node="human_review"
)

# 再度実行
for event in app.stream(None, config):
    print(event)
```

---

## 実用的な例

### Web API統合

```python
from flask import Flask, request, jsonify

app_flask = Flask(__name__)
graph_app = create_agent_graph()

@app_flask.route("/start", methods=["POST"])
def start_task():
    """タスク開始"""
    message = request.json["message"]
    thread_id = request.json["thread_id"]
    
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(content=message)]}
    
    try:
        for event in graph_app.stream(state, config):
            pass
        return jsonify({"status": "completed"})
    except:
        # interrupt発生
        snapshot = graph_app.get_state(config)
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            interrupt_data = snapshot.tasks[0].interrupts[0].value
            return jsonify({
                "status": "pending_approval",
                "data": interrupt_data,
                "thread_id": thread_id
            })

@app_flask.route("/approve", methods=["POST"])
def approve():
    """承認処理"""
    thread_id = request.json["thread_id"]
    approved = request.json["approved"]
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 状態を更新
    graph_app.update_state(
        config,
        {"approved": approved},
        as_node="human_review"
    )
    
    # 再開
    try:
        for event in graph_app.stream(None, config):
            pass
        return jsonify({"status": "completed"})
    except:
        return jsonify({"status": "error"})
```

---

## コンソール版での実装

```python
def main():
    app = create_agent_graph()
    config = {"configurable": {"thread_id": "1"}}
    
    state = {"messages": [HumanMessage(content="タスク")]}
    
    while True:
        completed = True
        
        try:
            # グラフ実行
            for event in app.stream(state, config):
                if event.get("final_answer"):
                    print("完了:", event["final_answer"])
                    return
            
        except Exception:
            # interrupt発生
            completed = False
            snapshot = app.get_state(config)
            
            if snapshot.tasks:
                task = snapshot.tasks[0]
                if task.interrupts:
                    # 承認プロンプト表示
                    interrupt_value = task.interrupts[0].value
                    print("承認が必要:", interrupt_value)
                    
                    user_input = input("承認しますか？ (y/n) > ")
                    
                    # 状態を更新して再開
                    app.update_state(
                        config,
                        {"approved": user_input.lower() == "y"},
                        as_node="human_review"
                    )
                    
                    state = None  # 既存の状態から続行
        
        if completed:
            break
```

---

## まとめ

| 項目 | `input()`方式 | `interrupt()`方式 |
|------|--------------|------------------|
| **コンソール** | ✅ 使える | ✅ 使える |
| **Web UI** | ❌ 使えない | ✅ 使える |
| **API統合** | ❌ 使えない | ✅ 使える |
| **非同期** | ❌ ブロック | ✅ 非ブロック |
| **状態永続化** | ❌ なし | ✅ あり |
| **複数ユーザー** | ❌ 難しい | ✅ 簡単 |

**推奨**: 本番環境では`interrupt()`を使用し、Web UIやAPIと統合するのがベストプラクティスです！

---

## 参考リンク

- [LangGraph Interrupts公式ドキュメント](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Human-in-the-Loop例](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
