import json
import os
from agents import Agent, ItemHelpers, Runner, AsyncOpenAI, OpenAIChatCompletionsModel , function_tool , RunContextWrapper , FunctionTool
from agents.run import RunConfig
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import Any
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
@function_tool(name_override="Read_File" )
def get_data(Context :RunContextWrapper[Any] , path: str , directory : str = None ) -> str:
    """Reads the contents of a file.

    Args:
        file_path: The path to the file to read.
        directory: The directory where the file is located. If not provided, it defaults to the current working directory.
    """
    return '<File Contents>'

@function_tool
def get_weather(city: str) -> str:
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return f"The weather in {city} is sunny"
fintess_agent = Agent(
    name = 'Fitness Agent',
    instructions='You are a Fitness Coach. You will help the user with their fitness goals, provide workout summaries, and answer questions about workouts.',
    model = model,
    tools = [get_data, get_weather],

)
def get_workout_summary(summary: str) -> str:
    return f"Workout Summary: {summary} , you burned 100 calories and worked out for 30 minutes."

class WorkoutParams(BaseModel):
    workout : str
    cals : int
    duration : int

async def get_workout_summary_tool(context: RunContextWrapper[Any], params: WorkoutParams) -> str:
    parsed = WorkoutParams.model_validate_json(params)
    return get_workout_summary(f"Workout: {parsed.workout}, Calories Burned: {parsed.cals}, Duration: {parsed.duration} minutes")

tracker = FunctionTool(
    name="Workout Tracker",
    description="Tracks workout details and provides a summary.",
    params_json_schema=WorkoutParams.model_json_schema(),
    on_invoke_tool=get_workout_summary_tool,
)

# def check_tools():
#     for tool in agent.tools:
#         print(f"Tool Name: {tool.name}")
#         print(f"Tool Description: {tool.description}")
#         print(json.dumps(tool.params_json_schema, indent=2))
#         print()

# check_tools()
# Mock library database (simulates a real database)
library_db = {
    "The Hobbit": {"available": True, "reserved_by": None},
    "1984": {"available": False, "reserved_by": "Alice"},
    "Pride and Prejudice": {"available": True, "reserved_by": None}
}

def manage_book_in_db(book_title: str, action: str, user_name: str = None) -> str:
    """Simulates database operations for checking or reserving a book."""
    if book_title not in library_db:
        return f"Error: {book_title} not found in the library."
    
    book = library_db[book_title]
    
    if action == "check":
        if book["available"]:
            return f"{book_title} is available."
        else:
            return f"{book_title} is not available, reserved by {book['reserved_by']}."
    
    elif action == "reserve":
        if not user_name:
            return "Error: User name required for reservation."
        if book["available"]:
            book["available"] = False
            book["reserved_by"] = user_name
            return f"{book_title} reserved for {user_name}."
        else:
            return f"Error: {book_title} is already reserved by {book['reserved_by']}."
    
    return "Error: Invalid action."

class BookArgs(BaseModel):
    book_title: str
    action: str  # Either "check" or "reserve"
    user_name: str | None = None  # Optional, required only for reserving

async def run_book_tool(ctx: RunContextWrapper[Any], args: str) -> str:
    """Processes book-related requests using the mock database."""
    try:
     print('Run[Debug] - run_book_tool called ')
     parsed = BookArgs.model_validate_json(args)
     return manage_book_in_db(parsed.book_title, parsed.action, parsed.user_name)
    except Exception as e:
        return f"Error processing request or Parsing Arguments: {str(e)}"

# Create the custom function tool
book_tool = FunctionTool(
    name="manage_book",
    description="Checks book availability or reserves a book in the library system",
    params_json_schema=BookArgs.model_json_schema(),
    on_invoke_tool=run_book_tool,
)

# Simulate using the tool in a chatbot (example usage)

library_chatbot = Agent(
        name="LibraryBot",
        instructions="Handle library queries by using the manage_book tool.",
        tools=[book_tool],
        model = model
)

async def run_agent():
    result = await Runner.run(starting_agent=library_chatbot, input='i want to reserve the book 1984 , my name is Abdullah and I am a regular memebr', )
    print(result.final_output)

asyncio.run(run_agent())


