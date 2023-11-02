import json
import openai
import os

# Set your OpenAI API key from an environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the locations and destinations for evaluation
evaluation_data = [
    {"location": "Ikeja", "destination": "National Stadium"},
    {"location": "Lekki", "destination": "Ojo"},
    {"location": "Masha", "destination": "Doyin"},
    {"location": "Surulere", "destination": "Badagry"},
    {"location": "Ikeja", "destination": "National Stadium"},
    {"location": "Lekki", "destination": "Ojo"},
    {"location": "Masha", "destination": "Doyin"},
    {"location": "Berger", "destination": "Festac"},
    {"location": "Computer Village", "destination": "Maryland"},
    {"location": "Iponri", "destination": "Aguda"},
    {"location": "Mile-2", "destination": "Iyana-Ipaja"},
    {"location": "Iyana Ipaja", "destination": "Costain"},
    {"location": "Epe", "destination": "Ebute-Meta"},
    {"location": "Costain", "destination": "LUTH"},
    {"location": "Idi-Araba", "destination": "Doyin"},
    {"location": "Admiralty Road, Lekki Phase 1", "destination": "Lekki, Phase 2"},
    {"location": "Ikeja", "destination": "Yaba"},
    {"location": "Ojo Barracks", "destination": "Trade-Fair"}
]

# Create an empty list to store evaluation examples
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
    # Add your own replies here
    expected_reply = ""

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
