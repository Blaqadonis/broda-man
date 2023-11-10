import openai
import wandb
import os
import json
from nltk.translate.bleu_score import sentence_bleu

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Import the fine-tuning model ID from the environment
model_id = os.environ["BRODAMAN_FINETUNE_MODEL_ID"]

# Initialize Weights & Biases
wandb.init(project="evaluating-new-model", entity="blaq")

# Export WandB entity, project, and API key to the environment
os.environ["WANDB_ENTITY"] = wandb.run.entity
os.environ["WANDB_PROJECT"] = wandb.run.project
os.environ["WANDB_API_KEY"] = wandb.api.api_key

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

    # Calculate the BLEU score
    reference = [expected_reply.split()]  # Convert expected output to a list of words
    candidate = model_output.split()

    # Compute BLEU score
    bleu_score = sentence_bleu(reference, candidate)

    # Log results to Weights & Biases
    wandb.log({
        "input_text": f"Location: {location}, Destination: {destination}",
        "expected_output": expected_reply,
        "model_output": model_output,
        "BLEU_score": bleu_score,
    })

    eval_results.append({
        "input_text": f"Location: {location}, Destination: {destination}",
        "expected_output": expected_reply,
        "model_output": model_output,
        "BLEU_score": bleu_score,
    })

# Save evaluation results to a JSON file
with open("evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)

# Finish Weights & Biases run
wandb.finish()

# Get the Weights & Biases link
wandb_link = wandb.run.get_url()

# Save the Weights & Biases link to an environment variable
os.environ["WANDB_LINK"] = wandb_link

print("Evaluation completed. Results logged to Weights & Biases.")
