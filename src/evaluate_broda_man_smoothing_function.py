import argparse
import openai
import wandb
import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate(model_id):
    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize Weights & Biases
    wandb.init(project="evaluating-broda-man", entity="blaq")

    # Load the evaluation dataset
    with open("evaluation_dataset.jsonl", "r") as jsonl_file:
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
            max_tokens=100  # Adjust max tokens as needed
        )

        model_output = response['choices'][0]['message']['content'].strip()

        # Calculate the BLEU score with smoothing
        reference = [expected_reply.split()]  # Convert expected output to a list of words
        candidate = model_output.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)

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

    # Calculate and log aggregated evaluation metrics (if needed)

    # Save evaluation results to a JSON file
    with open("evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    # Finish Weights & Biases run
    wandb.finish()

    print("Evaluation completed. Results logged to Weights & Biases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Broda-man model using BLEU scores.")
    parser.add_argument("--model-id", required=True, help="Fine-tuning model ID")
    args = parser.parse_args()
    
    evaluate(args.model_id)
