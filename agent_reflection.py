import os
import math
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def tool_calculator(expression: str):
    """Evaluates a mathematical expression safely."""
    try:
        allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

TOOLS = {"calculator": tool_calculator}

def agent_step_with_reflection(goal: str, max_steps: int = 3):
    """
    Runs an agent loop featuring a Reflection / Self-Correction step 
    before committing to a final output.
    """
    history = ""
    
    for step in range(1, max_steps + 1):
        print(f"\n--- [Step {step}/{max_steps}] ---")
        
        system_instruction = (
            "You are an autonomous self-learning agent equipped with a self-correction mechanism. "
            "Structure your response strictly as:\n"
            "THOUGHT: [Your reasoning]\n"
            "ACTION: [calculator: expression OR FINAL_ANSWER: your final response]"
        )

        prompt = f"{system_instruction}\n\nGoal: {goal}\nHistory:\n{history}\nNext Step:"

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        response_text = response.text
        print(f"🤖 Agent Response:\n{response_text}")

        # Check if agent reached a proposed final answer
        if "FINAL_ANSWER" in response_text:
            print("\n🔍 Running Reflection & Self-Correction Check...")
            
            # Extract the proposed answer content
            critique_prompt = (
                f"You are a strict Quality Assurance Critic. Review the following proposed final answer "
                f"against the original goal to check for mathematical errors, logic gaps, or omissions.\n\n"
                f"Original Goal: {goal}\n"
                f"Proposed Answer: {response_text}\n\n"
                f"Provide your critique strictly in this format:\n"
                f"CRITIQUE: [PASS or FAIL, followed by a short explanation]\n"
                f"REVISED_ANSWER: [If PASS, restate the final answer cleanly. If FAIL, provide the fully corrected final answer.]"
            )
            
            critique_response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=critique_prompt,
            )
            
            critique_text = critique_response.text
            print(f"💡 Reflection Result:\n{critique_text}")
            return critique_text

        # Handle tool actions if present
        if "ACTION:" in response_text and "calculator" in response_text:
            try:
                action_line = [line for line in response_text.split('\n') if "ACTION:" in line][0]
                action_content = action_line.split("ACTION:")[1].strip()
                if ":" in action_content:
                    _, tool_arg = action_content.split(":", 1)
                    tool_arg = tool_arg.strip()
                    print(f"🛠️ Executing calculator with arg: {tool_arg}")
                    observation = tool_calculator(tool_arg)
                    print(f"👀 Observation: {observation}")
                    history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: {observation}\n"
            except Exception as e:
                print(f"⚠️ Error parsing tool: {e}")
                break

    print("\n⚠️ Reached max steps without conclusion.")

if __name__ == "__main__":
    print("🧠 Initializing Self-Correction & Reflection Agent...")
    # A multi-step math problem designed to test accurate reasoning and critique
    test_goal = "Calculate the total cost if you buy 3 items priced at $45 each, apply a 20% discount to the subtotal, and then add a 5% sales tax on the discounted amount."
    agent_step_with_reflection(test_goal)