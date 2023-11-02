 # Broda-man

Broda-man is your cheerful assistant for beating traffic in Lagos State. Get real-time traffic advice and directions in Nigerian pidgin English.

![broda-man](https://github.com/Blaqadonis/broda-man/assets/100685852/df9e6402-c88f-4a56-96e9-6555e612547e)
## powered by 🅱🅻🅰🆀



## Table of Contents

- [About Broda-man](#about-broda-man)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- 


## About Broda-man

Broda-man is a chatbot designed to provide users with traffic-related information in Lagos State, Nigeria. It assists you in trying to find the quickest route from one location to another while navigating Lagos traffic even in the busiest of periods.

## Features

- Real-time traffic updates.
- Cheerful and entertaining responses in Nigerian pidgin English.
- Customized route suggestions to help you save time and avoid traffic jams.
- Easy-to-use and user-friendly interface.



## Getting Started

To get started with Broda-man, follow these simple steps:

1. Clone this repository to your local machine:
   ```shell
   git clone https://github.com/your-username/broda-man.git
2. Install the required dependencies. You'll need Python, so make sure you have it installed.
   Create a virtual environment with Conda.
   Edit ```<env_name>``` and ```3.X.X``` with environment name and python version of choice:
   ```shell
   conda create -n <env_name> python=3.X.X
3. Activate the virtual environment:
   ```shell
   conda activate <env_name>
4. Install the required Python packages:
   ```shell
   pip install -r requirements.txt
5. Start a job that i) generates synthetic dataset, uploads this dataset on the OpenAI platform, and finetunes a model with this uploaded dataset:
   ```shell
   python generate_and_train_broda_man.py
6. Routinely check this job's status:
   ```shell
   python job_status.py
7. Generate the evaluation dataset:
   ```shell
   python gen_eval_dataset.py
8. When the job's run is succeeded, go ahead and evaluate your model:
   ```shell
   python evaluate_broda_man.py
9. Finally, start Broda-man:
   ```shell
   python gradio_web_service_interface.py
10. Interact with [Broda man]([http://127.0.0.1:7860/]).



## Usage
1. Provide your current location and desired destination in the input fields.
2. Enjoy the cheerful responses and follow Broda-man's guidance to beat Lagos traffic.


## Contributing

We welcome contributions from the community to make Broda-man even better! If you have any suggestions, bug reports, or would like to contribute code, please follow our [contributing guidelines].



   
   
