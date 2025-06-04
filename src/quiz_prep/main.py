import os
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel , function_tool
from agents.run import RunConfig
from dotenv import load_dotenv
import asyncio
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(True)
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
#  Creatinng a tool to add with the agent 
@function_tool
def translate_in_german(text: str) -> str:
    ''' Translates the given text to German'''
    return f"Translated to German: {text} in German is 'Hallo! Wie geht es dir?'"


@function_tool
def get_weather(city: str) -> str:
    ''' 
    Returns the weather in the given city'''
    return f"The weather in {city} is sunny"

german_agent = Agent(
    name = 'German Translator',
    instructions = 'You have to translate the given text to German.',
    model = model,
    tools = [translate_in_german]
)
weather_agent = Agent(
    name='Weather Agent',
    instructions='You have to answer the given question about the weather. and when user asks about the weather in New York, you have to handoff the conversation to the New York Agent.',
    model=model,
    tools = [get_weather],
    handoffs=['New York Agent']  
)
new_york_weather_agent = Agent(
    name='New York Agent',
    instructions='You have to answer the given question about New York Weather.',
    model=model,
    tools = [get_weather]
)
async def run_agent():
    result = await Runner.run(starting_agent=german_agent , 
                              input='Hi ! How are you ?',)
    print(result.final_output)
def run_sync_agent():
    result = Runner.run_sync(starting_agent=german_agent, 
                             input='Hi ! How are you ?')
    print(result.final_output)    
async def run_async_agent():
    result = await Runner.run(starting_agent=german_agent, 
                                    input='Hi ! How are you ?')
    print(result)    
async def run_streamed_agent():
    result =  Runner.run_streamed(starting_agent=weather_agent,input =f'''What is the weather in New York? ''')
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            # print(event.data.delta, end="", flush=True)
            continue
        elif event.type == "run_item_stream_event":
           if event.item.type == "tool_call_item":
                print("-- Calling Weather APi")
           elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
           elif event.item.type == "message_output_item":
                print(f"-- Final Output:\n {ItemHelpers.text_message_output(event.item)}")
           else:
                pass  # Ignore other event types
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue

if __name__ == "__main__":
    operation = input("Enter the run mode (e.g., 'sync', 'async', 'streaming'): ").strip().lower()
    if operation not in ["sync", "async", "streaming"]:
        raise ValueError("Invalid operation. Please type 'sync', 'async', or 'streaming'.")
    if operation == 'sync':
        run_sync_agent()
    elif operation == 'async':
        asyncio.run(run_async_agent())
    elif operation == 'streaming':
        asyncio.run(run_streamed_agent())