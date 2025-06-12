import os
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel , function_tool
from agents.run import RunConfig
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX , prompt_with_handoff_instructions
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
set_tracing_disabled(True)
# set_trace_processors([OpenAIAgentsTracingProcessor()])
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
)
Technical_Support_Agent = Agent(
    name = "Technical Support Agent",
    instructions=prompt_with_handoff_instructions(prompt=f'''You are a technical support agent. You can assist with technical inquiries. '''),
    handoff_description='Handles technical inquiries and can transfer to other agents if needed.',
    model = model
  
)

billing_agent = Agent(
    name='Billing Support Agent',
    instructions=prompt_with_handoff_instructions(prompt = f'''You are a billing support agent. You can assist with billing inquiries.'''),
    handoff_description="Handles billing inquiries and can transfer to other agents if needed.",
    model=model,
    )

Triage_Agent = Agent(
    name = "Triage Agent",
    instructions=prompt_with_handoff_instructions(prompt = 'You are a customer service agent. You can assist with general inquiries.'
        'If the inquiry is related to billing, transfer it to the Billing Support Agent.'
        'If the inquiry is related to technical support, transfer it to the Technical Support Agent.'
        ),
    handoffs=[Technical_Support_Agent , billing_agent]    
)

async def run_support():
    result = await Runner.run(starting_agent=Triage_Agent, input="I have a problem with my account, can you help me?")
    print(result.final_output)

asyncio.run(run_support())