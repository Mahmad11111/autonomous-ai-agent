import os
import sqlite3
import math
from datetime import datetime
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DB_NAME = "agent_memory.db"

def init_db():
    """Initializes the SQLite database for agent memory and logs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Table for high-level goals and final answers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_text TEXT,
            final_answer TEXT,
            timestamp TEXT
        )
    """)
    
    # Table for detailed step-by-step execution logs (short-term & long-term trace)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id INTEGER,
            step_number INTEGER,
            thought TEXT,
            action TEXT,
            observation TEXT,
            FOREIGN KEY(goal_id) REFERENCES goals(id)
        )
    """)
    conn.commit()
    conn.close()

def log_goal_to_db(goal_text: str, final_answer: str) -> int:
    """Saves a completed goal and returns its goal_id."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO goals (goal_text, final_answer, timestamp) VALUES (?, ?, ?)", 
                   (goal_text, final_answer, timestamp))
    goal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return goal_id

def log_step_to_db(goal_id: int, step_num: int, thought: str, action: str, observation: str):
    """Saves individual step telemetry into memory."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO execution_logs (goal_id, step_number, thought, action, observation)
        VALUES (?, ?, ?, ?, ?)
    """, (goal_id, step_num, thought, action, observation))
    conn.commit()
    conn.close()

def get_past_memories():
    """Retrieves past completed goals from long-term storage."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT goal_text, final_answer, timestamp FROM goals ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()
    return rows

def tool_calculator(expression: str):
    """Evaluates a mathematical expression safely."""
    try:
        allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs}
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

TOOLS = {"calculator": tool_calculator}

def run_memory_agent(goal: str):
    """Runs the agent with persistent memory storage enabled."""
    init_db()
    
    # Check if we have relevant past memories to inform current run
    past_memories = get_past_memories()
    memory_context = ""
    if past_memories:
        memory_context = "\nPast Agent Experience / Memories:\n"
        for g, a, t in past_memories:
            memory_context += f"- Goal: '{g}' | Result: '{a}' ({t})\n"

    print(f"🧠 Agent starting goal with {len(past_memories)} records in long-term memory.")
    
    history = ""
    final_result = "Incomplete"
    goal_id = log_goal_to_db(goal, "Pending")
    
    for step in range(1, 3):
        print(f"\n--- [Step {step}/2] ---")
        
        system_instruction = (
            "You are an autonomous memory-backed agent. Structure your response strictly as:\n"
            "THOUGHT: [Your reasoning]\n"
            "ACTION: [calculator: expression OR FINAL_ANSWER: your final response]"
        )

        prompt = f"{system_instruction}\n{memory_context}\nGoal: {goal}\nHistory:\n{history}\nNext Step:"

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        response_text = response.text
        print(f"🤖 Agent Response:\n{response_text}")

        thought, action, observation = "N/A", "N/A", "N/A"
        
        # Simple line parser
        for line in response_text.split('\n'):
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()

        if "FINAL_ANSWER" in response_text:
            final_result = response_text
            log_step_to_db(goal_id, step, thought, action, "Goal Completed")
            break

        if "calculator:" in action:
            tool_arg = action.split("calculator:")[1].strip()
            observation = tool_calculator(tool_arg)
            print(f"👀 Observation: {observation}")
            history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: {observation}\n"
            log_step_to_db(goal_id, step, thought, action, observation)

    # Update goal final status in database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE goals SET final_answer = ? WHERE id = ?", (final_result, goal_id))
    conn.commit()
    conn.close()
    print(f"\n💾 Successfully saved session to SQLite database (`{DB_NAME}`).")

if __name__ == "__main__":
    test_goal = "what was the last question I asked?"
    run_memory_agent(test_goal)