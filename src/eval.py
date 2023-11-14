import json
import os
import openai
import getpass
import wandb
from nltk.translate.bleu_score import sentence_bleu

# OpenAI API key for generating the evaluation dataset
if os.getenv("OPENAI_API_KEY") is None:
    if any(['VSCODE' in x for x in os.environ.keys()]):
        print('Please enter the password in the VS Code prompt at the top of your VS Code window!')
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI Key from: https://platform.openai.com/account/api-keys\n")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

# Set up WandB project and run for generating the evaluation dataset
wandb.init(project="evaluation-dataset")

# Locations and destinations for evaluation
evaluation_data = [
    {"location": "Ikeja", "destination": "National Stadium"},
    # Add more locations and destinations as needed
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

# Log evaluation examples to WandB
wandb.log({"evaluation_examples": evaluation_examples})

# Save the evaluation dataset to a JSON file
with open("evaluation_dataset.jsonl", "w") as jsonl_file:
    for example in evaluation_examples:
        json.dump(example, jsonl_file)
        jsonl_file.write("\n")

# Log the evaluation dataset to WandB
artifact = wandb.Artifact(name="evaluation_dataset", type="dataset")
artifact.add_file("evaluation_dataset.jsonl")
wandb.run.log_artifact(artifact)

print(f"Evaluation dataset saved to 'evaluation_dataset.jsonl' with {len(evaluation_examples)} examples.")

# Hardcoded initial part of the model ID
base_model_id = "ft:gpt-3.5-turbo-0613:personal:broda-man:"

# Import the fine-tuning model ID suffix from the environment
model_id_suffix = os.environ["BRODAMAN_FINETUNE_MODEL_SUFFIX"]
#model_id_suffix = "8KYjZe8R"
# Concatenate the initial part and the extracted model ID suffix
model_id = base_model_id + model_id_suffix

# Initialize Weights & Biases for model evaluation
run = wandb.init(project="evaluating-new-model", entity="blaq")

# Retrieve the evaluation dataset from WandB
artifact = wandb.run.use_artifact("evaluation_dataset:latest")
artifact_dir = artifact.download()

# Load the evaluation dataset
with open(os.path.join(artifact_dir, "evaluation_dataset.jsonl"), "r") as jsonl_file:
    evaluation_data = [json.loads(line) for line in jsonl_file]

# Evaluate the model
eval_results = []

for example in evaluation_data:
    location = example["messages"][0]["content"].split(":")[1].strip()
    destination = example["messages"][1]["content"].split(":")[1].strip()
    expected_reply = example["expected_reply"]

    # Generate a response from the model using ChatCompletion
    user_messages = [
        {"role": "system", "content": "You are Broda-man, the Lagos state traffic bot. You assist users who want to beat traffic Lagos at all costs, by providing them with routes with less traffic when they provide you with their location and destination details. You respond strictly and only in Nigerian pidgin language. You are often cheerful too."},
        {"role": "user", "content": f"Location: {location}, Destination: {destination}"}
    ]
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=user_messages,
        max_tokens=50  # Adjust max tokens as needed
    )

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

# Log results to Weights & Biases
wandb.log({"evaluation_results": eval_results})

# Finish Weights & Biases run
wandb.finish()

# Get the Weights & Biases link
wandb_link = run.get_url()

# Save the Weights & Biases link to an environment variable
os.environ["WANDB_LINK"] = wandb_link

print("Evaluation completed. Results logged to Weights & Biases.")
