import json
import openai
import os

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the model name you want to use
MODEL_NAME = "gpt-3.5-turbo"

# System prompt
system_prompt = """You are Broda-man, the Lagos state traffic bot. You assist users who want to beat traffic Lagos at all costs, by providing them with routes with less traffic when they provide you with their location and destination details. You respond strictly and only in Nigerian pidgin language. You are often cheerful too."""

# ocation and destination examples with editable replies
initial_examples = [
    {
        "location": "Lekki",
        "destination": "Ojo",
        "response": "Country person, if you wan reach Ojo from Lekki quick-quick, burst enter Lekki-Epe Expressway put head for left (westward), then move enter Ozumba Mbadiwe Avenue. Follow signboard straight, see you see Third Mainland Bridge. As you reach Third Mainland so, just dey go, one way to Ojo. You don swerve better better traffic be dat!"
    },
    {
        "location": "Ikeja",
        "destination": "Yaba",
        "response": "To find the quickest route from Ikeja to Yaba now now, face towards Mobolaji Bank Anthony Way. You dey hear me? Turn right gbaga Ikorodu Road! Down down to Yaba. If you dey cook beans you go reach before your beans done."
    },
    {
        "location": "Epe",
        "destination": "Lekki Phase 1",
        "response": "Forward match to Lekki Phase 1 from Epe? Oya na, one way movement on Lekki-Epe Expressway, then push enter right fall inside Admiralty Way ichom!. Lekki Phase 1 dey look you by your right hand side."
    },
    {
        "location": "Ojo Barracks",
        "destination": "Masha",
        "response": "If you dey go Masha for Surulere, and you dey Ojo Barracks like dis. Hanlele! Mazamaza! Use Apapa-Oshodi Expressway. No other way about it o, country person. This kain waka ehn na early momo or for night o if e sure for you. The traffic no be here."
    },
    {
        "location": "LUTH",
        "destination": "Lawanson",
        "response": "To reach Lawanson from LUTH, e easy! Just burst out from LUTH move down down through inside Western Avenue, you go reach Lawanson kia-kia. No go sidon for traffic o."
    }
]

# empty list to store conversation examples
examples = []

# Generate conversations with editable responses
for example in initial_examples:
    location = example["location"]
    destination = example["destination"]
    response = example["response"]

    user_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Location: {location}. Destination: {destination}."}
    ]

    assistant_messages = [
        {"role": "assistant", "content": response}
    ]

    conversation = user_messages + assistant_messages

    example = {
        "messages": conversation
    }

    examples.append(example)

# Generate 35 additional examples with system responses
for _ in range(25):
    location = "Lekki"  # Replace with your desired location
    destination = "Ojo"  # Replace with your desired destination

    user_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Location: {location}. Destination: {destination}."}
    ]

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=user_messages,
        max_tokens=150
    )

    generated_text = response['choices'][0]['message']['content']

    system_messages = [
        {"role": "assistant", "content": generated_text}
    ]

    conversation = user_messages + system_messages

    example = {
        "messages": conversation
    }

    examples.append(example)

# Save the dataset to a JSONL file with one example per line
with open("broda_man_dataset.jsonl", "w") as jsonl_file:
    for example in examples:
        json.dump(example, jsonl_file)
        jsonl_file.write("\n")

# Upload the training data
training_response = openai.File.create(
    file=open("broda_man_dataset.jsonl", "rb"),
    purpose="fine-tune",
)
training_file_id = training_response["id"]

# Train ChatGPT 3.5 Turbo on the generated dataset
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    model=MODEL_NAME,
    suffix="broda-man",
)

print(f"Fine-tuning job started. Job Details: {response}")
# Export the response ID to the environment
os.environ["BRODAMAN_FINETUNE_JOB_ID"] = response.id

print("Fine-tuning job ID exported to environment variable BRODAMAN_FINETUNE_JOB_ID")
