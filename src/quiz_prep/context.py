import json
import os
from agents import Agent, ItemHelpers, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, Tool , function_tool , RunContextWrapper , FunctionTool, handoff , set_trace_processors , GuardrailFunctionOutput, InputGuardrailTripwireTriggered , TResponseInputItem , input_guardrail , output_guardrail , OutputGuardrailTripwireTriggered , AgentHooks , RunHooks
from agents.run import RunConfig
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing_extensions import Any
import asyncio
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dataclasses import dataclass
from langsmith.wrappers import OpenAIAgentsTracingProcessor
from agents.extensions.visualization import draw_graph
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


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
    tracing_disabled=True,
    workflow_name='Travel Agent Workflow',
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
    try:
        parsed = FlightBookingTool.model_validate_json(args)
        if parsed.cost > context.context.budget:
          return f"Insufficient budget. You have ${context.context.budget}, but the flight costs ${parsed.cost}."
        context.context.budget -= parsed.cost
        return f"Flight to {context.context.destination} booked for ${parsed.cost}. Your remaining budget is ${context.context.budget}."
    except Exception as e:
        print(f"Validation error: {e}") if isinstance(e, ValidationError) else print(f"Error: {e}")
        return f"Error processing request : {str(e)}"
    
 #  Tool context manager to dynamically enable or disable the tool based on the budget    # 
def tool_context_manager(ctx: RunContextWrapper[TravelerInfo], agent: Agent[TravelerInfo]):
    if(ctx.context.budget > 1000):
        print(f"Tool context manager enabled for {ctx.context.username} with budget ${ctx.context.budget}.")
        return True
    else:
        print(f"Tool context manager disabled for {ctx.context.username} with budget ${ctx.context.budget}.")
        return False
        
Book_Flight_tool = FunctionTool(
    name="BookFlight",
    description="Book a flight to a destination with a specified cost.",
    params_json_schema=FlightBookingTool.model_json_schema(),
    on_invoke_tool=BookFlight,
)
#  Dummy Function tool to check fucntionality
@function_tool
def get_weather(city: str) -> str:
    '''
    Args:
        city (str): The name of the city to get the weather for.
    Returns:
    str: A string describing the weather in the specified city.    
    '''
    return f"The weather in {city} is sunny with a high of 25°C."

# def dynamic_instructions(context: RunContextWrapper[TravelerInfo], agent: Agent[TravelerInfo]) -> str:
#     return f"""
#     You are a travel agent assisting {context.context.username} with their travel plans to {context.context.destination}. Their current budget is ${context.context.budget}. Your role is to help with flight bookings and weather information using exactly two tools: 'manage_travel' and 'get_weather'. No other tools exist, so do not invent or combine tools (e.g., avoid calling 'BookFlightget_weather').
#     - Use the 'manage_travel' tool to book a flight when the user requests a flight booking (e.g., 'Book a flight' or 'Reserve a $200 flight'). This tool requires a numeric cost in USD (e.g., 200.0) and checks the user's budget.
#     - Use the 'get_weather' tool to check the weather when the user asks about weather in a specific city (e.g., 'What’s the weather in Makkah?'). This tool requires a city name.
#     For queries with multiple tasks (e.g., 'Book a flight and check the weather'), handle each task separately:
#     1. First, call 'manage_travel' to book the flight if a cost is provided.
#     2. Then, call 'get_weather' to check the weather if a city is mentioned.
#     If the query is unclear or doesn’t match either tool, respond with a clarification request (e.g., 'Please specify the flight cost or city for weather'). Always validate that your tool choice matches the query intent and is one of the two available tools.
#     """


def dynamicInstructions_bookingflight(context: RunContextWrapper[TravelerInfo], agent: Agent[TravelerInfo]) -> str:
    return f"""
    {RECOMMENDED_PROMPT_PREFIX}
    You are a Flight Booking Agent that handles flight booking inquiries. You have a budget of ${context.context.budget} to work with.
    - Use the 'BookFlight' tool to book a flight when the user requests a flight booking (e.g., 'Book a flight' or 'Reserve a $200 flight'). This tool requires a numeric cost in USD (e.g., 200.0) and checks the user's budget.
    - If the query is unclear or doesn’t match the tool, respond with a clarification request (e.g., 'Please specify the flight cost or city for weather'). Always validate that your tool choice matches the query intent and is one of the available tools.
    if the query is related to weather, transfer it to the Weather Agent.
    """
#  Input GuardRail Agent
class Is_Israel(BaseModel):
    is_israel: bool
    reason : str

input_guardrail_agent = Agent(
    name='Input Guardrail Agent',
    instructions="Check if the user is trying to book a flight to Israel.",
    model=model,
    output_type=Is_Israel
)
@input_guardrail
async def check_israel(context: RunContextWrapper[TravelerInfo] , agent : Agent ,  input : str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(
        starting_agent=input_guardrail_agent,
        input=input,
        context=context.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_israel
    )
#  Input GuardRail Agent
class Is_Professional(BaseModel):
    is_not_professional: bool
    reason : str

output_guardrail_agent = Agent(
    name='Output Guardrail Agent',
    instructions="Check if the response from the agent is professional , travel related and correct. ",
    model=model,
    output_type=Is_Professional
)
@output_guardrail
async def check_output(context: RunContextWrapper[TravelerInfo] , agent : Agent ,  input : str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(
        starting_agent=output_guardrail_agent,
        input=input,
        context=context.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_not_professional
    )

#  Run Hooks for the workflow
class RunHooks(RunHooks):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.name = "Travel Agent Run Hooks"
    async def on_agent_start(self, context: RunContextWrapper[TravelerInfo], agent: Agent):
        self.counter += 1
        print(f"Run Hooks started for agent {agent.name} for user {context.context.username}. Invocation count: {self.counter}")    
    async def on_tool_start(self, context: RunContextWrapper[TravelerInfo], agent: Agent, tool: Tool):
        self.counter += 1
        print(f"Run Hooks Tool {tool.name} invoked by agent {agent.name}. Invocation count: {self.counter}")
    async def on_tool_end(self, context: RunContextWrapper[TravelerInfo], agent: Agent, tool: Tool, result: str):
        print(f"Run Hooks Tool {tool.name} completed by agent {agent.name}. Result: {result}")
    async def on_agent_end(self, context: RunContextWrapper[TravelerInfo], agent: Agent, output: Any):
        self.counter += 1
        print(f"Run Hooks ended for agent {agent.name} for user {context.context.username}. Invocation count: {self.counter}")

Run = RunHooks()
#  Creating Agent Lifecycle Hooks
class Hooks(AgentHooks):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.name = "Travel Agent"
    async def on_start(self, context : RunContextWrapper[TravelerInfo], agent : Agent):
        self.counter += 1
        print(f"Agent {self.name} started for user {context.context.username}. Invocation count: {self.counter}")    
    async def on_tool_start(self, context : RunContextWrapper[TravelerInfo], agent : Agent, tool : Tool):
        self.counter += 1
        print(f"Tool {tool.name} invoked by agent {agent.name}. Invocation count: {self.counter}") 
    async def on_tool_end(self, context : RunContextWrapper[TravelerInfo], agent : Agent, tool : Tool, result : str):
        print(f"Tool {tool.name} completed by agent {agent.name}. Result: {result}")   
    async def on_end(self, context : RunContextWrapper[TravelerInfo], agent : Agent , output : Any):
        print(f"Agent {agent.name} ended for user {context.context.username}. Output: {output}. Total invocations: {self.counter}")

travel_agent_hooks = Hooks()


Flight_Booking_agent = Agent[TravelerInfo](
    name='Flight Booking Agent',
    instructions=dynamicInstructions_bookingflight,
    model=model,
    tools=[Book_Flight_tool],
    input_guardrails=[check_israel],
    # output_guardrails=[check_output],
    # model_settings=ModelSettings(
    #     tool_choice="required",
    # )
    # hooks=travel_agent_hooks,
)

Weather_agent = Agent(
    name='Weather Agent',
    instructions=f'''{RECOMMENDED_PROMPT_PREFIX}You are a Weather Agent that handles weather inquiries. You can answer questions about the weather in different cities. Use the 'get_weather' tool to provide weather information.If the query is related to flight booking, transfer it to the Flight Booking Agent.''',
    model=model,
    tools=[get_weather],
    # input_guardrails=[check_israel],
    # output_guardrails=[check_output],
    # model_settings=ModelSettings(
    #     tool_choice="required",
    # )
    # hooks=travel_agent_hooks,
)
triage_agent = Agent(
    name="Triage Agent",
    instructions=f'''{RECOMMENDED_PROMPT_PREFIX} You are Travel Agent that handles general travel inquiries , 
      If the query is related to flight booking, transfer it to the Flight Boking Agent,
      If the query is related to weather, transfer it to the Weather Agent.
      If the query inlcudes both the flight booking and weather, transfer it to the Flight Boking Agent and then to the Weather Agent.''',
    model = model,
    input_guardrails=[check_israel],
    # output_guardrails=[check_output],
)
class HandoffData(BaseModel):
    reason : str
def on_handoffs(ctx:RunContextWrapper[TravelerInfo], input_data:HandoffData):
    print(f"Handoff Reason: {input_data.reason} ")
triage_agent.handoffs =[
    handoff(agent=Flight_Booking_agent, on_handoff=on_handoffs, input_type=HandoffData  ),
    handoff(agent=Weather_agent, on_handoff=on_handoffs, input_type=HandoffData  )
]
Flight_Booking_agent.handoffs =[
    handoff(agent=Weather_agent, on_handoff=on_handoffs, input_type=HandoffData  )
]

Weather_agent.handoffs =[
    handoff(agent=Flight_Booking_agent, on_handoff=on_handoffs, input_type=HandoffData  )
]
async def run_agent():
    traveler_info = TravelerInfo(
        username="John Doe",
        userid="123456",
        budget=560.0,
        destination="Makkah, Saudi Arabia"
    )
    try : 
     result = await Runner.run(starting_agent=triage_agent, context=traveler_info, input="Hi ! I want to book a flight to Makkah for $200 , and also tell me the weather in Makkah?", )
     print(result.final_output)
    #  print(result.to_input_list)
    #  result = Runner.run_streamed(
    #     starting_agent=travel_agent,
    #     context=traveler_info,
    #     input = "Book a Flight for $500?",
    #     hooks = Run
    # )
    #  async for event in result.stream_events():
    #      if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
    #             print(event.data.delta, end="", flush=True)
             
    # new_input = result.to_input_list() + [{"role": "user", "content": "What's the remaining budget Now?"}]
    # result = await Runner.run(travel_agent, new_input)
    # print(result.final_output)
    except Exception as e:
        if isinstance(e, InputGuardrailTripwireTriggered):
            print(f"Input Guardrail Tripwire Triggered: {e.guardrail_result.output.output_info}")
        elif isinstance(e, OutputGuardrailTripwireTriggered):
            print(f"Output Guardrail Tripwire Triggered: {e.guardrail_result.output.output_info}")
        else:
            print(f"An error occurred: {e}")
#  Drawing the agent graph
# draw_graph(travel_agent , filename="agent_graph.png")
asyncio.run(run_agent())