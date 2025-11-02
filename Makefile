.PHONY: help install run-agent run-streamlit test clean lint format

# デフォルトターゲット
help:
	@echo "利用可能なコマンド:"
	@echo "  make install        - 仮想環境を作成して依存パッケージをインストール"
	@echo "  make run-agent      - agent_with_hitl.py を実行"
	@echo "  make run-streamlit  - Streamlit アプリを起動"
	@echo "  make test           - テストを実行"
	@echo "  make lint           - コードの静的解析（Pylint）"
	@echo "  make format         - コードのフォーマット（Black）"
	@echo "  make clean          - 仮想環境とキャッシュを削除"
	@echo "  make activate       - 仮想環境のアクティベート方法を表示"

# 仮想環境の作成とパッケージインストール（uv使用）
install:
	@echo "🔧 仮想環境を作成中..."
	uv venv
	@echo "📦 依存パッケージをインストール中..."
	uv pip install -r requirements.txt
	@echo "✅ インストール完了！"
	@echo "💡 仮想環境をアクティベートするには: source .venv/bin/activate"

# agent_with_hitl.py を実行
run-agent:
	@echo "🤖 エージェントを起動中..."
	uv run python agent_with_hitl.py

# Streamlit アプリを起動
run-streamlit:
	@echo "🚀 Streamlit アプリを起動中..."
	uv run streamlit run streamlit_app.py

# agents.py を使ったアプリを起動
run-agents:
	@echo "🤖 agents.py のエージェントを起動中..."
	uv run streamlit run streamlit_app.py

# テストを実行
test:
	@echo "🧪 テストを実行中..."
	uv run pytest tests/ -v

# テスト（Bedrock接続確認）
test-bedrock:
	@echo "☁️  AWS Bedrock 接続テスト中..."
	uv run python test_bedrock.py

# Pylint でコード解析
lint:
	@echo "🔍 コードを解析中..."
	uv run pylint *.py

# Black でコードフォーマット
format:
	@echo "✨ コードをフォーマット中..."
	uv run black *.py

# グラフの可視化
visualize:
	@echo "📊 グラフを可視化中..."
	uv run python visualize_graph.py

# 仮想環境とキャッシュの削除
clean:
	@echo "🧹 クリーンアップ中..."
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ クリーンアップ完了！"

# 仮想環境のアクティベート方法を表示
activate:
	@echo "💡 仮想環境をアクティベートするには:"
	@echo "   source .venv/bin/activate"
	@echo ""
	@echo "📝 非アクティベート化するには:"
	@echo "   deactivate"

# 環境変数の確認
check-env:
	@echo "🔍 環境変数を確認中..."
	@if [ -f .env ]; then \
		echo "✅ .env ファイルが存在します"; \
		echo ""; \
		echo "📋 .env の内容:"; \
		cat .env; \
	else \
		echo "⚠️  .env ファイルが見つかりません"; \
		echo "💡 .env.example をコピーして設定してください:"; \
		echo "   cp .env.example .env"; \
	fi

# 依存パッケージの更新（uv使用）
update:
	@echo "🔄 依存パッケージを更新中..."
	uv pip install --upgrade -r requirements.txt
	@echo "✅ 更新完了！"

# requirements.txt の生成（uv使用）
freeze:
	@echo "📝 requirements.txt を生成中..."
	uv pip freeze > requirements.txt
	@echo "✅ requirements.txt を更新しました"


# 開発用サーバーの起動（Streamlit + ホットリロード）
dev:
	@echo "🔥 開発モードで起動中（ホットリロード有効）..."
	uv run streamlit run streamlit_app.py --server.runOnSave=true
