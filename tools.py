from langchain_experimental.llms.ollama_functions import OllamaFunctions

model = OllamaFunctions(model="gemma:2b", format="json")

model = model.bind_tools(
    tools=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. Kathmandu, Nepal",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    function_call={"name": "get_current_weather"},
)


from langchain_core.messages import HumanMessage

model.invoke("what is the weather in Boston?")