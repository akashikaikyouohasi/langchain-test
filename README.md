# LangGraph エージェント with Human-in-the-Loop

複数のツール、Human-in-the-Loop（interrupt使用）、構造化された出力を組み合わせたLangGraphエージェントの実装例です。
AWS Bedrockを使用してClaude 3.5 Sonnetで動作します。

## 特徴

1. **複数のツール**: Web検索、計算、情報取得などのツールを使用
2. **Human-in-the-Loop (interrupt版)**: 
   - LangGraphの`interrupt()`機能を使用
   - グラフを一時停止して外部から再開可能
   - Web UI、API、Slackなどと統合可能
3. **構造化された出力**: Pydanticモデルを使用して、最終的な回答を構造化
4. **AWS Bedrock**: Claude 3.5 Sonnetを使用（OpenAI不要）
5. **状態永続化**: チェックポイントで会話状態を保持

## セットアップ

### 1. 依存関係のインストール

uvを使用した環境構築：

```bash
# 仮想環境を作成
uv venv

# 仮想環境を有効化
source .venv/bin/activate

# パッケージをインストール
uv pip install -r requirements.txt
```

### 2. AWS認証情報の設定

AWS CLIで認証情報を設定するか、`.env`ファイルで設定します：

**方法1: AWS CLIを使用（推奨）**

```bash
aws configure
```

**方法2: .envファイルを使用**

```bash
cp .env.example .env
# .envファイルを編集してAWS認証情報を設定
```

`.env`ファイルの内容：
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
```

## 使用方法

```bash
# 仮想環境を有効化
source .venv/bin/activate

# メイン実装（interrupt版）を実行
python agent_with_hitl.py

# シンプル版を実行
python simple_agent.py
```

### interrupt版の動作

1. グラフがツール実行前に**自動的に一時停止**
2. コンソールで承認/拒否を選択
3. 承認すればツール実行、拒否すればフィードバックを送信
4. 状態が保存されるため、後から再開も可能

## 実装の詳細

### グラフの構造

```
[開始] → [agent] → 条件分岐
                    ├─ [tools] → [agent] (通常のツール実行)
                    ├─ [human_review] → interrupt → 承認待ち (重要な操作)
                    │                    ├─ 承認 → [tools]
                    │                    └─ 拒否 → [agent]
                    └─ [finalize] → [終了] (構造化された出力)
```

### interrupt機能の仕組み

**従来の方式 (`input()`)**
- ❌ コンソールでブロッキング
- ❌ Web UIと統合不可
- ❌ 状態の永続化なし

**新しい方式 (`interrupt()`)**
- ✅ グラフを一時停止
- ✅ 外部システムから再開可能
- ✅ 状態をチェックポイントに保存
- ✅ Web UI、API、Slackなどと統合可能

```python
# interrupt()でグラフを停止
approval_data = interrupt({
    "type": "human_review",
    "tool_calls": [...],
    "message": "承認が必要です"
})

# 外部から状態を更新して再開
app.update_state(config, {"approved": True}, as_node="human_review")
```

### ツール

- `search_web`: Web検索を実行
- `calculator`: 数式を計算
- `get_current_info`: 特定のトピックの情報を取得

### Human-in-the-Loop

計算ツール（`calculator`）が呼び出される際、人間の承認が必要になります。ユーザーは以下を選択できます：

- **承認**: ツールの実行を許可
- **拒否**: フィードバックを提供してエージェントに再考させる

### 構造化された出力

最終的な出力は`FinalAnswer` Pydanticモデルとして構造化されます：

```python
{
    "summary": "タスクの要約",
    "findings": ["発見1", "発見2"],
    "calculations": {"式1": 結果1},
    "confidence": 0.95,
    "sources": ["ソース1", "ソース2"]
}
```

## カスタマイズ

### 新しいツールを追加

```python
@tool
def my_custom_tool(param: str) -> str:
    """ツールの説明"""
    # 実装
    return "結果"

# ツールリストに追加
tools = [search_web, calculator, get_current_info, my_custom_tool]
```

### Human-in-the-Loopのトリガー条件を変更

`should_continue`関数を編集して、どのツールで人間の承認が必要かを変更できます：

```python
def should_continue(state: AgentState) -> Literal["tools", "human_review", "finalize"]:
    # ...
    if "my_custom_tool" in tool_names:
        return "human_review"
    # ...
```

### 構造化された出力のスキーマを変更

`FinalAnswer`クラスを編集して、必要なフィールドを追加・変更できます：

```python
class FinalAnswer(BaseModel):
    summary: str
    # 新しいフィールドを追加
    recommendation: str = Field(description="推奨事項")
```

## 例

### 例1: 計算タスク

```
タスク: 123 × 456 を計算して、その結果について教えてください

→ エージェントがcalculatorツールを使用しようとする
→ Human-in-the-Loopが発動
→ ユーザーが承認
→ 計算実行
→ 構造化された結果を出力
```

### 例2: 情報収集タスク

```
タスク: Pythonの最新バージョンについて調べてください

→ エージェントがsearch_webツールを使用
→ 自動的に実行（承認不要）
→ 構造化された結果を出力
```

## 注意事項

- AWS Bedrockへのアクセス権限が必要です
- Claude 3.5 Sonnetモデルへのアクセスが有効になっている必要があります
- `calculator`ツールはevalを使用しているため、本番環境では安全な実装に置き換えてください
- Web検索ツールはサンプル実装です。実際のAPI（SerpAPIなど）と統合する必要があります

## AWS Bedrockのセットアップ

1. AWSコンソールでBedrockサービスにアクセス
2. モデルアクセスページでClaude 3.5 Sonnetを有効化
3. 適切なIAMロールとポリシーを設定

必要なIAMポリシー例：
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
    }
  ]
}
```
