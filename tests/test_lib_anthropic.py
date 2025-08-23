#!/usr/bin/env python3
"""
直接使用 Anthropic Python 库的测试代码
"""

import os
from anthropic import Anthropic


model_name = "claude-3-5-sonnet-20241022"

def test_anthropic_with_base_url(base_url=None, stream=False):
    """使用自定义 base_url 测试 Anthropic API（用于测试代理）"""
    
    client = Anthropic(api_key=anthropic_api_key, base_url=base_url)
    
    try:
        if stream:
            response = client.messages.create(
                model=model_name,
                max_tokens=50,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": "Hello! This is a test through proxy."}
                ],
                stream=True,
            )
            
            print("📡 流式响应:", end=" ")
            for chunk in response:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    print(chunk.delta.text, end="")
            print(f"\n✅ 流式响应测试完成 (base_url: {base_url})")
        else:
            response = client.messages.create(
                model=model_name,
                max_tokens=50,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": "Hello! This is a test through proxy."}
                ]
            )
            
            print(f"✅ Anthropic 通过代理调用成功 (base_url: {base_url})")
            print(f"回复: {response.content[0].text}")
        
    except Exception as e:
        print(f"❌ Anthropic 代理调用失败: {e}")


anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_base_url = "http://localhost:8899/anthropic"


def main():
    if not anthropic_api_key:
        print("❌ 错误: 未设置 ANTHROPIC_API_KEY 环境变量")
        return
    
    print("🧪 开始 Anthropic 代理测试...")
    
    # 测试直接调用（不使用代理）
    print("\n1. 测试直接调用 Anthropic API:")
    test_anthropic_with_base_url()
    
    # 测试通过代理调用
    print(f"\n2. 测试通过代理调用（非流式）:")
    test_anthropic_with_base_url(anthropic_base_url)
    
    # 测试通过代理调用（流式）
    print(f"\n3. 测试通过代理调用（流式）:")
    test_anthropic_with_base_url(anthropic_base_url, stream=True)


if __name__ == "__main__":
    main()