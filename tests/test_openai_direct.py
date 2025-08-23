#!/usr/bin/env python3
"""
直接使用 OpenAI Python 库的测试代码
"""

import os
from openai import OpenAI


def test_openai_with_base_url(base_url=None, stream=False):
    """使用自定义 base_url 测试 OpenAI API（用于测试代理）"""
    
    client = OpenAI(api_key=openai_api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello! This is a test through proxy."}
            ],
            max_tokens=50,
            temperature=0.7,
            stream=stream,
        )
        
        if stream:
            print("📡 流式响应:", end=" ")
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
            print(f"\n✅ 流式响应测试完成 (base_url: {base_url})")
        else:
            print(f"✅ OpenAI 通过代理调用成功 (base_url: {base_url})")
            print(f"回复: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ OpenAI 代理调用失败: {e}") 

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = "http://localhost:8080/openai/v1"

def main():
    
    test_openai_with_base_url()
    test_openai_with_base_url(openai_base_url)
    test_openai_with_base_url(openai_base_url, stream=True)


if __name__ == "__main__":
    main()