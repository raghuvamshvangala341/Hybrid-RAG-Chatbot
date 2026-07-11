from llm.groq_client import GroqClient


client = GroqClient()

response, metrics = client.generate(

    system_prompt="You are a helpful assistant.",

    user_prompt="What is Artificial Intelligence?"

)

print("\nResponse:\n")

print(response)

print("\nMetrics:\n")

print(metrics)