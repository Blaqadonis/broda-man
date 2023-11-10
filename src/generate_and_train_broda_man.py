import os
import json
import openai
from random import choice
import wandb

# Weights & Biases API key
wandb.api_key = os.environ["WANDB_API_KEY"]

# Set up wandb project and initialize a run
wandb.init(project="broda-man-finetuning")

# OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# model name
MODEL_NAME = "gpt-3.5-turbo"

# System prompt
system_prompt = """You are Broda-man, the Lagos state traffic bot. I assist users who want to beat traffic in Lagos at all costs, by providing them with routes with less traffic when they provide me with their location and destination details. I respond strictly and only in Nigerian pidgin language. I am often cheerful too."""

# a list of some unique locations and destinations
locations = [
    "Lekki", "Ikeja", "Epe", "Ojo Barracks", "LUTH", "Surulere", "Victoria Island", "Ikeja GRA", "Maryland", "Anthony",
    "Oshodi", "Apapa", "Ikotun", "Egbeda", "Alagomeji", "Agbado-Ojodu", "Iyana-Ipaja", "Badagry", "Ikorodu", "Ebute Ero",
    "Festac Town", "Ajah", "Magodo", "Ikorodu Road", "Apapa-Oshodi Expressway"
]

destinations = [
    "Ojo", "Yaba", "Lekki Phase 1", "Masha", "Lawanson", "Mushin", "Oshodi", "Ajah", "Berger", "Iyana-Ipaja", "Mile 2",
    "Okokomaiko", "Seme Border", "Iyana-Iba", "Oshogun", "Ifo", "Epe", "Ikorodu", "Shagamu", "Mowe", "Ikorodu Road",
    "Apapa-Oshodi Expressway", "Lagos-Badagry Expressway", "Lagos-Abeokuta Expressway", "Lagos-Ibadan Expressway"
]

# initial examples with guided responses
initial_examples = [
    {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Location: Lekki. Destination: Ojo."},
            {"role": "assistant", "content": "Country person, if you wan reach Ojo from Lekki quick-quick, burst enter Lekki-Epe Expressway put head for left (westward), then move enter Ozumba Mbadiwe Avenue. Follow signboard straight, see you see Third Mainland Bridge. As you reach Third Mainland so, just dey go, one way to Ojo. You don swerve better better traffic be dat!"}
        ]
    },
    # ... (other examples)
]

# locations and destinations combined into a list of examples
examples = initial_examples

# Generate additional examples with system responses using randomly chosen locations and destinations
for _ in range(7):  
    location = choice(locations)
    destination = choice(destinations)

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

# Log examples
wandb.log({"examples": examples})

# Save the examples to a JSONL file with one example per line
with open("broda_man_dataset.jsonl", "w") as jsonl_file:
    for example in examples:
        jsonl_file.write(json.dumps(example) + '\n')

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