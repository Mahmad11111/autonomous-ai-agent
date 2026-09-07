import os
import math
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Autonomous AI Agent API with Logging", version="2.0")

DB_NAME = "agent_api_logs.db"

def init_api_db():
    """Initializes the database to store API execution traces and logs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal TEXT,
            status TEXT,
            final_output TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_api_db()

# --- Tools Definition ---
def tool_calculator(expression: str):
    try:
        allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs}
        return str(eval(expression, {"__builtins__" : {}}, allowed_names))
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

def tool_file_writer(file_data: str):
    try:
        if "|" not in file_data:
            return "Error: Expected format 'filename.txt | content'"
        filename, content = file_data.split("|", 1)
        with open(filename.strip(), "w", encoding="utf-8") as f:
            f.write(content.strip())
        return f"Successfully wrote content to {filename.strip()}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

TOOLS = {
    "calculator": tool_calculator,
    "file_writer": tool_file_writer
}

class AgentRequest(BaseModel):
    goal: str
    max_steps: int = 3

@app.get("/")
def home():
    return {"status": "Agent API with Logging is online", "phase": "Phase 3 - Day 8 Complete"}

@app.post("/run-agent")
def run_agent_endpoint(req: AgentRequest):
    history = ""
    steps_trace = []
    final_response_text = "Max steps reached without completion"
    status = "Incomplete"
    
    for step in range(1, req.max_steps + 1):
        system_instruction = (
            "You are an autonomous API agent equipped with 'calculator' and 'file_writer' tools.\n"
            "Structure your response strictly as:\n"
            "THOUGHT: [Your reasoning]\n"
            "ACTION: [tool_name: argument OR FINAL_ANSWER: your final response]"
        )

        prompt = f"{system_instruction}\n\nGoal: {req.goal}\nHistory:\n{history}\nNext Step:"

        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            response_text = response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

        step_info = {"step": step, "agent_response": response_text}

        if "FINAL_ANSWER" in response_text:
            step_info["status"] = "Completed"
            steps_trace.append(step_info)
            final_response_text = response_text
            status = "Success"
            break

        if "ACTION:" in response_text:
            try:
                action_line = [line for line in response_text.split('\n') if "ACTION:" in line][0]
                action_content = action_line.split("ACTION:")[1].strip()
                
                if ":" in action_content:
                    tool_name, tool_arg = action_content.split(":", 1)
                    tool_name = tool_name.strip().lower()
                    tool_arg = tool_arg.strip()
                    
                    if tool_name in TOOLS:
                        observation = TOOLS[tool_name](tool_arg)
                        step_info["tool_executed"] = tool_name
                        step_info["observation"] = observation
                        history += f"\nStep {step}:\nResponse: {response_text}\nOBSERVATION: {observation}\n"
                    else:
                        step_info["observation"] = f"Tool '{tool_name}' not found."
            except Exception as e:
                step_info["error"] = str(e)

        steps_trace.append(step_info)

    # Save log to SQLite database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO api_logs (goal, status, final_output, timestamp) VALUES (?, ?, ?, ?)",
                   (req.goal, status, final_response_text, timestamp))
    conn.commit()
    conn.close()

    return {
        "goal": req.goal,
        "status": status,
        "steps": steps_trace,
        "final_output": final_response_text
    }

@app.get("/logs")
def get_logs():
    """Endpoint to view all past API task logs from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, goal, status, final_output, timestamp FROM api_logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    logs = [{"id": r[0], "goal": r[1], "status": r[2], "final_output": r[3], "timestamp": r[4]} for r in rows]
    return {"total_logs": len(logs), "logs": logs}