import os
import wandb

# Load the saved WandB run
run = wandb.init()

# Save the WandB run URL to the environment variable and GITHUB_OUTPUT file
WANDB_LINK = run.get_url()
with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    print(f'WANDB_LINK={WANDB_LINK}', file=f)
