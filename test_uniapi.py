from openai import OpenAI
import base64
import re
import os

# 初始化客户端
uniapi_key = os.getenv("UNIAPI_KEY")
client = OpenAI(api_key=uniapi_key, base_url="https://api.uniapi.io/v1")

# 调用接口生成图片
response = client.chat.completions.create(
    model="gemini-3.1-flash-image-preview",
    messages=[{"role": "user", "content": "请生成一张可爱的小海獭在海洋中游泳的图片"}],
    stream=False,
    extra_body={
        "modalities": ["text", "image"]  # 指定输出模态包含图片
    },
)

# 获取响应内容
content = response.choices[0].message.images[0]["image_url"]["url"]
print("响应内容：", content[:100], "...")

# 提取 Base64 图片数据
# 响应格式为: ![image](data:image/png;base64,xxxx)
match = re.search(r"data:image/(\w+);base64,([A-Za-z0-9+/=]+)", content)
if match:
    img_format = match.group(1)
    img_data = match.group(2)

    # 解码并保存图片
    image_bytes = base64.b64decode(img_data)
    output_file = f"output.{img_format}"

    with open(output_file, "wb") as f:
        f.write(image_bytes)

    print(f"图片已保存到 {output_file}")
else:
    print("未找到图片数据，响应内容：", content)
