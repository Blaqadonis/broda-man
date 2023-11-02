import openai
import os

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Import the fine-tuning job ID from the environment
job_id = os.environ["BRODAMAN_FINETUNE_JOB_ID"]

# Check the status of the fine-tuning job
job = openai.FineTuningJob.retrieve(id=job_id)

# Print the status
print("Job Status:", job.status)
