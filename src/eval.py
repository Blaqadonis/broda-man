import json
import os
import openai
import getpass
from nltk.translate.bleu_score import sentence_bleu

# OpenAI API key for generating the evaluation dataset
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Ensure that the OpenAI API key is set
if not openai.api_key:
    if any(['VSCODE' in x for x in os.environ.keys()]):
        print('Please enter the password in the VS Code prompt at the top of your VS Code window!')
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI Key from: https://platform.openai.com/account/api-keys\n")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

assert openai.api_key.startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

# Locations and destinations for evaluation
evaluation_data = [
    {"location": "Ikeja", "destination": "National Stadium"},
    {"location": "Lekki", "destination": "Eko Hotel"},
    {"location": "Surulere", "destination": "University of Lagos"},
    {"location": "Victoria Island", "destination": "Lekki Conservation Centre"},
    {"location": "Yaba", "destination": "Murtala Muhammed Airport"},
    {"location": "Ikorodu", "destination": "The Palms Shopping Mall"},
    {"location": "Ajao Estate", "destination": "Ikeja City Mall"},
    {"location": "Ojota", "destination": "Computer Village"},
    {"location": "Apapa", "destination": "Tin Can Island Port"},
    {"location": "Oshodi", "destination": "Mushin Market"},
    {"location": "Agege", "destination": "New Garage"},
    {"location": "Badagry", "destination": "Badagry Heritage Museum"},
    {"location": "Epe", "destination": "Epe Fish Market"},
    {"location": "Ikoyi", "destination": "Ikoyi Club 1938"},
    {"location": "Makoko", "destination": "Makoko Floating School"},
    {"location": "Obalende", "destination": "Tafawa Balewa Square"},
    {"location": "Ijora", "destination": "Iganmu Industrial Estate"},
    {"location": "Alimosho", "destination": "Egbeda Akowonjo Roundabout"},
    {"location": "Gbagada", "destination": "Anthony Village"},
    {"location": "Festac", "destination": "Festac Town Link Bridge"},
    {"location": "Isolo", "destination": "Aswani Market"},
    {"location": "Igando", "destination": "Igando General Hospital"},
    {"location": "Oshodi-Isolo", "destination": "Cele Expressway"},
    {"location": "Amuwo-Odofin", "destination": "Apple Junction"},
    {"location": "Agbado", "destination": "Agbado Crossing"},
    {"location": "Alakuko", "destination": "Ijaiye Market"},
    {"location": "Egbeda", "destination": "MicCom Golf Course"},
    {"location": "Ilupeju", "destination": "Coker Market"},
    {"location": "Magodo", "destination": "Magodo Phase II"},
]

# An empty list to store evaluation examples
evaluation_examples = []

# Generate conversations for evaluation
for data in evaluation_data:
    location = data["location"]
    destination = data["destination"]

    user_messages = [
        {"role": "user", "content": f"Location: {location}."},
        {"role": "user", "content": f"Destination: {destination}."}
    ]

    # Modify the expected assistant reply for each location-destination pair
    expected_reply = f"Modify this with the expected reply for {location} to {destination}"

    evaluation_example = {
        "messages": user_messages,
        "expected_reply": expected_reply
    }

    evaluation_examples.append(evaluation_example)

# Save the evaluation dataset to a JSON file
with open("evaluation_dataset.jsonl", "w") as jsonl_file:
    for example in evaluation_examples:
        json.dump(example, jsonl_file)
        jsonl_file.write("\n")

print(f"Evaluation dataset saved to 'evaluation_dataset.jsonl' with {len(evaluation_examples)} examples.")


# Import the fine-tuning model ID suffix from the environment
model_id = os.getenv('BRODAMAN_FINETUNE_MODEL_ID', '')
# model_id_suffix = "8MMdTENP"

# Initialize OpenAI for model evaluation
openai.api_key = openai.api_key  # Use the updated API key
openai.organization = None  # Remove the organization parameter if not using an organization

# Evaluate the model
eval_results = []

for example in evaluation_examples:
    location = example["messages"][0]["content"].split(":")[1].strip()
    destination = example["messages"][1]["content"].split(":")[1].strip()
    expected_reply = example["expected_reply"]

    # Generate a response from the model using ChatCompletion
    user_messages = [
        {"role": "system",
         "content": "You are Broda-man, the Lagos state traffic bot. You assist users who want to beat traffic Lagos at all costs, by providing them with routes with less traffic when they provide you with their location and destination details. You respond strictly and only in Nigerian pidgin language. You are often cheerful too."},
        {"role": "user", "content": f"Location: {location}, Destination: {destination}"}
    ]
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=user_messages,
        max_tokens=60)

    model_output = response['choices'][0]['message']['content'].strip()

    # Print reference and candidate
    print("Reference:", expected_reply)
    print("Candidate:", model_output)

    # Calculate the BLEU score only if the reference is not empty
    if expected_reply:
        reference = [expected_reply.split()]  # Convert expected output to a list of words
        candidate = model_output.split()

        # Compute BLEU score
        bleu_score = sentence_bleu(reference, candidate)

        eval_results.append({
            "input_text": f"Location: {location}, Destination: {destination}",
            "expected_output": expected_reply,
            "model_output": model_output,
            "BLEU_score": bleu_score,
        })
    else:
        print("Warning: Expected reply is empty. Skipping BLEU score calculation.")

# Save evaluation results to a JSON file
with open("evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)

print("Evaluation completed. Results logged to 'evaluation_results.json'.")
