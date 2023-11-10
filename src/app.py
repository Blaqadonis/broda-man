import gradio as gr
import openai
import os
import threading
import time
import sys
import json

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import the fine-tuning model ID from the environment
model_id = os.getenv("BRODAMAN_FINETUNE_MODEL_ID", "ft:gpt-3.5-turbo-0613:personal:broda-man:8J4pz8Md")

# Function to generate completions
def generate_completion(location, destination):
    """Generates a completion using the fine-tuned model."""
    
    # Define the system prompt for your fine-tuned model
    system_prompt = """You are Broda-man, the Lagos state traffic bot. You assist users who want to beat traffic in Lagos at all costs, by providing them with routes with less traffic when they provide you with their location and destination details. You respond strictly and only in Nigerian pidgin language. You are often cheerful too."""

    # Construct messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Location: {location}\nDestination: {destination}"},
    ]

    # Generate a completion using your fine-tuned model
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
    )

    # Strip the response of whitespace
    return response["choices"][0]["message"]["content"].strip()

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_completion,
    inputs=[
        gr.Textbox(lines=5, label="Location", placeholder="Country person, which side you dey?"),
        gr.Textbox(lines=5, label="Destination", placeholder="Where you dey go?"),
    ],
    outputs="text",
    title="Ask Broda-man! Your Friendly Transport Route Bot",
    live=False,  # Set live to False to use share
)

# Define a function to run Gradio and return the share URL
def run_gradio():
    share_url = iface.share()
    print(f"Gradio interface is live at: {share_url}")
    sys.stdout.flush()

# Create a thread to run Gradio
gradio_thread = threading.Thread(target=run_gradio)

# Start the Gradio thread
gradio_thread.start()

# Add additional processing or wait logic if needed
# ...

# Save the share URL to a file for later use
with open("gradio_share_url.json", "w") as url_file:
    json.dump({"share_url": iface.share()}, url_file)

# Provide a delay to keep the workflow running (adjust as needed)
for _ in range(30):
    print("Waiting...")
    sys.stdout.flush()
    time.sleep(10)

# Wait for the Gradio thread to finish
gradio_thread.join()

print("Workflow completed.")
sys.stdout.flush()
