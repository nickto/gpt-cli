import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.8,
    n=1,
    messages=[
        {
            "role": "system",
            "content": "You a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Read 2 CSV files using Pandas and merge them on id column, then compute summary statistics for each of the numerical columns?",
        },
    ],
)

for i, choice in enumerate(completion.choices):
    reply = choice.message["content"]
    print("====================")
    print(f"Choice {i:d}:\n")
    print(reply)
