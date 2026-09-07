import os
from dotenv import load_dotenv
from google import genai

# Load our environment variables from the .env file
load_dotenv()

# Look up the variable name we defined in our .env file
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key not found! Please check your .env file.")

# Initialize the official Google GenAI client
client = genai.Client(api_key=api_key)

def ask_agent(prompt: str):
    """
    Sends a prompt to the Gemini model and returns its response.
    This acts as the basic 'thought' function for our agent.
    """
    try:
        # Using gemini-2.5-flash (or gemini-3.6-flash depending on your account availability)
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with agent brain: {str(e)}"

# Quick test if run directly
if __name__ == "__main__":
    print("🤖 Agent Brain Initialized Successfully!")
    test_prompt = "Hello! Introduce yourself as a self-learning autonomous agent in one sentence."
    print(f"Prompt: {test_prompt}")
    print(f"Agent Response: {ask_agent(test_prompt)}")