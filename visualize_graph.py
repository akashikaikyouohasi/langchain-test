"""
可視化ツール: グラフの構造をMermaid形式で出力

agent_with_hitl.pyのグラフ構造を可視化します。
"""

def visualize_graph():
    """グラフの構造をMermaid形式で出力"""
    mermaid = """
```mermaid
graph TD
    Start([開始]) --> Agent[Agent Node]
    
    Agent --> Decision{条件分岐}
    
    Decision -->|ツール呼び出し<br/>かつ重要な操作| HumanReview[Human Review Node]
    Decision -->|ツール呼び出し<br/>通常の操作| Tools[Tools Node]
    Decision -->|ツール呼び出しなし| Finalize[Finalize Node]
    
    HumanReview --> HumanDecision{ユーザーの判断}
    HumanDecision -->|承認| Tools
    HumanDecision -->|拒否/フィードバック| Agent
    
    Tools --> Agent
    
    Finalize --> End([終了])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style HumanReview fill:#FFD700
    style Agent fill:#87CEEB
    style Tools fill:#DDA0DD
    style Finalize fill:#98FB98
```

## グラフの説明

### ノード

1. **Agent Node**: LLMがメッセージを処理し、次のアクションを決定
2. **Human Review Node**: 重要な操作の前に人間の承認を求める
3. **Tools Node**: ツールを実行
4. **Finalize Node**: 構造化された最終出力を生成

### フロー

1. 開始 → Agent
2. Agent → 条件分岐:
   - 重要なツール呼び出し → Human Review
   - 通常のツール呼び出し → Tools
   - ツール呼び出しなし → Finalize
3. Human Review:
   - 承認 → Tools
   - 拒否 → Agent（フィードバック付き）
4. Tools → Agent（結果を反映）
5. Finalize → 終了

### Human-in-the-Loop トリガー

以下の条件で人間の承認が必要になります：
- `calculator`ツールの使用時
- （カスタマイズ可能）

### 構造化された出力

Finalize Nodeで生成される`FinalAnswer`の構造:

```python
{
    "summary": "タスクの要約",
    "findings": ["発見事項1", "発見事項2", ...],
    "calculations": {"計算式": 結果, ...},
    "confidence": 0.0~1.0,
    "sources": ["情報源1", "情報源2", ...]
}
```
"""
    print(mermaid)


if __name__ == "__main__":
    visualize_graph()
