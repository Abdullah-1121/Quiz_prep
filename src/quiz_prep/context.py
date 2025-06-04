import json
import os
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel , function_tool , RunContextWrapper , FunctionTool , set_trace_processors
from agents.run import RunConfig
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import Any
import asyncio
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dataclasses import dataclass
from langsmith.wrappers import OpenAIAgentsTracingProcessor
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
set_trace_processors([OpenAIAgentsTracingProcessor()])
@dataclass
class TravelerInfo:
    username: str
    userid : str
    budget: float
    destination: str

class FlightBookingTool(BaseModel):
    cost: float
async def BookFlight(context: RunContextWrapper[TravelerInfo] , args : str) -> str:
    parsed = FlightBookingTool.model_validate_json(args)
    context.context.budget -= parsed.cost
    return f"Flight to {context.context.destination} booked for ${parsed.cost}. Your remaining budget is ${context.context.budget}."

manage_travel = FunctionTool(
    name="BookFlight",
    description="Book a flight to a destination with a specified cost.",
    params_json_schema=FlightBookingTool.model_json_schema(),
    on_invoke_tool=BookFlight,
)

def dynamic_instructions(context: RunContextWrapper[TravelerInfo] , agent : Agent[TravelerInfo]) -> str:
    return f"You are a travel agent. You will help {context.context.username} with their travel plans, including booking flights and managing their budget. Their current budget is ${context.context.budget} and they are planning to travel to {context.context.destination}."

travel_agent = Agent[TravelerInfo](
    name='Travel Agent',
    instructions=dynamic_instructions,
    model=model,
    tools=[manage_travel],
)

async def run_agent():
    traveler_info = TravelerInfo(
        username="John Doe",
        userid="123456",
        budget=1000,
        destination="New York"
    )
    result = await Runner.run(starting_agent=travel_agent, context=traveler_info, input="I want to book a flight to New York for $500.")
    print(result.final_output)

asyncio.run(run_agent())