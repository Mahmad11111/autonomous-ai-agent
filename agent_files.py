import os
import math
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Custom Tools Definition ---

def tool_calculator(expression: str):
    """Evaluates a mathematical expression safely."""
    try:
        allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs}
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

def tool_file_writer(file_data: str):
    """
    Writes content to a file. 
    Format expected: 'filename.txt | content to write'
    """
    try:
        if "|" not in file_data:
            return "Error: File writer expects format 'filename.txt | content'"
        filename, content = file_data.split("|", 1)
        filename = filename.strip()
        content = content.strip()
        
        # Ensure safety: write within the current directory
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote content to {filename}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def tool_file_reader(filename: str):
    """Reads content from a text file."""
    try:
        filename = filename.strip()
        if not os.path.exists(filename):
            return f"Error: File '{filename}' not found."
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Unified Tool Registry
TOOLS = {
    "calculator": tool_calculator,
    "file_writer": tool_file_writer,
    "file_reader": tool_file_reader
}

def run_file_agent(goal: str, max_steps: int = 3):
    """Runs an agent capable of using math and file management tools."""
    history = ""
    
    for step in range(1, max_steps + 1):
        print(f"\n--- [Step {step}/{max_steps}] ---")
        
        system_instruction = (
            "You are an autonomous agent equipped with multiple tools: 'calculator', 'file_writer', and 'file_reader'.\n"
            "- To use calculator, format action as: calculator: expression\n"
            "- To use file_writer, format action as: file_writer: filename.txt | content here\n"
            "- To use file_reader, format action as: file_reader: filename.txt\n"
            "- When finished, use: FINAL_ANSWER: your final response\n\n"
            "Structure your response strictly as:\n"
            "THOUGHT: [Your reasoning]\n"
            "ACTION: [tool_name: argument OR FINAL_ANSWER: text]"
        )

        prompt = f"{system_instruction}\n\nGoal: {goal}\nHistory:\n{history}\nNext Step:"

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        response_text = response.text
        print(f"🤖 Agent Response:\n{response_text}")

        if "FINAL_ANSWER" in response_text:
            print("\n✅ Agent completed the task successfully!")
            return response_text

        # Parse and execute tool calls
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
                        print(f"👀 Observation:\n{observation}")
                        
                        history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: {observation}\n"
                    else:
                        observation = f"Error: Tool '{tool_name}' not found."
                        print(f"👀 Observation: {observation}")
                        history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: {observation}\n"
            except Exception as e:
                print(f"⚠️ Error executing action: {e}")
                break

    print("\n⚠️ Reached max steps without conclusion.")

if __name__ == "__main__":
    print("📂 Initializing Multi-Tool File Agent...")
    # Give the agent a multi-step task involving file creation and math computation
    test_goal = "tell me who is babar azam and give his details in a new txt file named as babar.txt"
    run_file_agent(test_goal)