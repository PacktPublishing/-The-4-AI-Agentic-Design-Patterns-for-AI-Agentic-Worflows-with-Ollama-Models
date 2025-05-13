import autogen
from autogen import AssistantAgent

# This is a simple example of how to use the AssistantAgent class with the Llama model.
# Setup API key. Add your own API key to config file or environment variable
config_list = [
    {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
]

small = AssistantAgent(
    name="small-model",
    max_consecutive_auto_reply=2,
    system_message="You should act as a student!",
    llm_config={
        "config_list": config_list,
        "temperature": 1,
    },
)

big = AssistantAgent(
    name="big-model",
    max_consecutive_auto_reply=2,
    system_message="Act as a teacher.",
    llm_config={
        "config_list": config_list,
        "temperature": 1,
    },
)


res = big.initiate_chat(small, message="Who are you?")

print(res)
