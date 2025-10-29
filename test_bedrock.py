"""
AWS Bedrock接続テスト

このスクリプトでAWS Bedrockに接続できるか確認します。
"""

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

def test_bedrock_connection():
    """AWS Bedrock接続をテスト"""
    print("AWS Bedrock接続テスト")
    print("="*50)
    
    # 環境変数の確認
    model_id = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    print(f"モデル: {model_id}")
    print(f"リージョン: {region}")
    print()
    
    try:
        # ChatBedrockインスタンスを作成
        llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={
                "temperature": 0,
                "max_tokens": 100
            }
        )
        
        # 簡単なテストメッセージ
        print("テストメッセージを送信中...")
        response = llm.invoke("こんにちは！簡単な自己紹介をしてください。")
        
        print("\n✅ 接続成功！")
        print("\nレスポンス:")
        print("-"*50)
        print(response.content)
        print("-"*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\n確認事項:")
        print("  1. AWS CLIで認証情報が設定されているか確認")
        print("     → aws configure")
        print("  2. Bedrockサービスへのアクセス権限があるか確認")
        print("  3. Claude 3.5 Sonnetモデルへのアクセスが有効か確認")
        print("     → AWSコンソール > Bedrock > Model access")
        print(f"  4. 正しいリージョン({region})を使用しているか確認")
        return False


if __name__ == "__main__":
    test_bedrock_connection()
