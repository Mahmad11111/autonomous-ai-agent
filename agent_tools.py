import os
import json
import math
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define available tools that the agent can choose to "act" upon
def tool_calculator(expression: str):
    """Evaluates a mathematical expression safely."""
    try:
        # Safe evaluation of basic math
        allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

# Map tool names to actual functions
TOOLS = {
    "calculator": tool_calculator
}

def run_react_agent(goal: str, max_steps: int = 3):
    """
    Runs a multi-step ReAct loop allowing the agent to reason, 
    call tools, observe results, and reach a final answer.
    """
    history = ""
    
    for step in range(1, max_steps + 1):
        print(f"\n--- [Step {step}/{max_steps}] ---")
        
        system_instruction = (
            "You are an autonomous self-learning agent with access to tools. "
            "You must structure your response strictly as:\n"
            "THOUGHT: [Your reasoning]\n"
            "ACTION: [tool_name: argument OR FINAL_ANSWER: your final response]\n"
            "CONTENT: [Details if needed]"
        )

        prompt = f"{system_instruction}\n\nGoal: {goal}\nHistory:\n{history}\nNext Step:"

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        
        response_text = response.text
        print(response_text)
        
        # Check if agent is done
        if "FINAL_ANSWER" in response_text:
            print("\n✅ Agent reached final answer!")
            return response_text

        # Simple parser for tool action execution
        if "ACTION:" in response_text:
            try:
                action_line = [line for line in response_text.split('\n') if "ACTION:" in line][0]
                action_content = action_line.split("ACTION:")[1].strip()
                
                if ":" in action_content:
                    tool_name, tool_arg = action_content.split(":", 1)
                    tool_name = tool_name.strip().lower()
                    tool_arg = tool_arg.strip()
                    
                    if tool_name in TOOLS:
                        print(f"🛠️ Executing tool [{tool_name}] with arg: {tool_arg}")
                        observation = TOOLS[tool_name](tool_arg)
                        print(f"👀 Observation: {observation}")
                        
                        # Append to history for the next loop iteration
                        history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: {observation}\n"
                    else:
                        history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: Error - Tool '{tool_name}' not found.\n"
            except Exception as e:
                print(f"⚠️ Error parsing action: {e}")
                break

    print("\n⚠️ Reached max steps without explicit final answer.")

if __name__ == "__main__":
    print("🚀 Starting Multi-Step Tool-Enabled ReAct Agent...")
    # Give it a task that benefits from a calculation tool
    complex_goal = "Calculate the square root of 144 plus 50, then multiply by 2."
    run_react_agent(complex_goal)