#Generate dataset using openai

from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

def get_ai_generated_data():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ No OpenAI API key - using placeholder data")
        return [{"instruction": "What are your technical skills?", "response": "Python, data analysis"}]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = "Create 2 interview Q&A pairs for a software developer in JSON format. Output only JSON."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    content = response.choices[0].message.content

    # If model returns Markdown-style JSON block
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1]

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data.get("examples", [])
        elif isinstance(data, list):
            return data
        else:
            print("⚠️ Unexpected JSON format:", type(data))
            return []
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        print("Raw content:", content)
        return []

# Example call
get_ai_generated_data()

data = get_ai_generated_data()
print(data)