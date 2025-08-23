
# "/openai/v1/chat/completions" => ('openai', 'v1/chat/completions')
def convert_url(url: str):
    result = tuple(url.lstrip("/").split("/", 1))
    return result 

# https://api.openai.com/v1 -> https://api.openai.com
# https://api.openai.com/v1/ -> https://api.openai.com
# https://api.moonshot.cn/anthropic/v1/ -> https://api.moonshot.cn/anthropic
def get_base_url(url: str) -> str:
    if url.endswith("/"):
        url = url[:-1]
    if url.endswith("/v1"):
        url = url[:-3]
    return url