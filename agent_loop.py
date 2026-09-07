import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key not found! Please check your .env file.")

client = genai.Client(api_key=api_key)

def react_agent_step(goal: str, history: str = ""):
    """
    Executes a single ReAct (Reason + Act) step with explicit configuration.
    """
    system_instruction = (
        "You are an autonomous self-learning agent operating in a ReAct loop. "
        "For any given goal, you must structure your response strictly as follows:\n"
        "THOUGHT: [Your internal reasoning about what to do next]\n"
        "ACTION: [The action or tool call you want to execute, or 'FINAL_ANSWER' if complete]\n"
        "CONTENT: [The output or final answer corresponding to the action]"
    )

    prompt = f"{system_instruction}\n\nGoal: {goal}\nPrevious History:\n{history}\n\nNext Step:"

    try:
        # Calling gemini-3.6-flash with standard generation config
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error in ReAct loop: {str(e)}"

if __name__ == "__main__":
    print("🔄 Initializing ReAct Agent Loop...")
    test_goal = "HOW CRICKET IS PLAYED?"
    print(f"Goal: {test_goal}\n")
    
    step_result = react_agent_step(test_goal)
    print("Agent ReAct Output:")
    print(step_result)