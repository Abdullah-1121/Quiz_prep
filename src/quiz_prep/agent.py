import os
from agents import Agent, AgentOutputSchema, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel , handoff,Handoff,function_tool , AgentOutputSchemaBase, set_trace_processors , RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX , prompt_with_handoff_instructions
from agents.run import RunConfig
from dotenv import load_dotenv
import asyncio
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dataclasses import dataclass
from langsmith.wrappers import OpenAIAgentsTracingProcessor
from typing import Any
from agents.extensions import handoff_filters
from pydantic import BaseModel
# set_tracing_disabled(True)
# set_trace_processors([OpenAIAgentsTracingProcessor()])
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
set_trace_processors([OpenAIAgentsTracingProcessor()])
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
    tracing_disabled=True,
    workflow_name='Quiz Prep Workflow',
)
@dataclass
class Jokes:
    jokes : dict[int , str]

@dataclass
class FinalOutput:
    jokes : list[str]
@function_tool
def get_weather(city: str) -> str:
    """Fetch the weather for a given location.

    Args:
        city: The city to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return f"The weather in {city} is sunny" 

class HandoffData(BaseModel):
    reason : str
async def on_handoffs(ctx : RunContextWrapper[None] , input_data :HandoffData  ):
    print(f"Handoff Reason: {input_data.reason} ")
Technical_Support_Agent = Agent(
    name = "Technical Support Agent",
    instructions=f'''{RECOMMENDED_PROMPT_PREFIX}You are a technical support agent. You can assist with technical inquiries. ''',
    model = model ,
    handoff_description='Handles technical inquiries and can transfer to other agents if needed.'
)
Billing_Support_Agent = Agent(
    name='Billing Support Agent',
    instructions=f'''{RECOMMENDED_PROMPT_PREFIX}You are a billing support agent. You can assist with billing inquiries. ''',
    model=model,
    handoff_description='Handles billing inquiries and can transfer to other agents if needed.'
)  

basic_agent = Agent(
    name='Personal Assistant',
    instructions=f'''{RECOMMENDED_PROMPT_PREFIX}You are a Customer Support Agent that handles general inquiries ,' \
    'and If the query is related to the technical issues, Transfer  to the Technical Support Agent, and if the query is related to billing issues,Transfer to the Billing Support Agent.''',
    model=model,
    tools=[get_weather],
    # output_type=Jokes
)
basic_agent.handoffs = [
    handoff(agent=Technical_Support_Agent, on_handoff=on_handoffs, input_type=HandoffData ),
    handoff(agent=Billing_Support_Agent, on_handoff=on_handoffs, input_type=HandoffData , 
            input_filter=handoff_filters.remove_all_tools ),]



# basic_agent.output_type = AgentOutputSchema(Jokes, strict_json_schema=False) 
#  On using an external provider it wil give bad request 400 (for example in gemini)
async def run_basic_agent():
    result = await Runner.run(starting_agent=basic_agent, input='How can you help me?', )
    print(result)
    # print(result.final_output_as(FinalOutput , raise_if_incorrect_type=True))
    # result= Runner.run_streamed(starting_agent=basic_agent, input='Your website is not working fine , it is taking very much time to open ', )
    # async for event in result.stream_events():
    #     if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
    #         print(event.data.delta, end="", flush=True)
    #     elif event.type == "run_item_stream_event":
    #         if event.item.type == "tool_call_item":
    #             print("-- Calling Weather APi")
    #             print(result.current_turn)
    #         if event.item.type == "tool_call_output_item":    
    #             print(f"-- Tool output: {event.item.output}")
    #             print(result.current_turn)
    #     elif event.type == "agent_updated_stream_event":
    #         print(f"Agent updated: {event.new_agent.name}")
    #         print(result.current_turn)

asyncio.run(run_basic_agent())
