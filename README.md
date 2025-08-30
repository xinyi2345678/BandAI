**Getting Started
Please access our streamlit app at [https://bandai-pfgqstfbdsrvdpfbkjelns.streamlit.app/](url)

To run locally with your own AI,
1. Unzip BandAI.zip to extract all relevant files.

2. pip install -r requirements.txt

3. Insert your own openAI API key and model name inside AI_main.py

4. streamlit run BandAI.py

To use this application, you will need to provide your own OpenAI API key. This key allows the application to make requests to OpenAI's models on your behalf.

Configuration Steps

Follow these instructions to get your project running.
1. Obtain Your OpenAI API Key

First, you need to get a secret API key from the OpenAI Platform.

Sign up or Log in: Go to the OpenAI Platform website at platform.openai.com.[1] If you don't have an account, you'll need to sign up.

Navigate to API Keys: Once logged in, click on your profile icon or name in the top-right corner and select "View API keys" from the dropdown menu.[2]

Create a New Secret Key: Click on the "+ Create new secret key" button.[2]

Name Your Key: A pop-up will appear asking you to name your key. It's good practice to give it a descriptive name (e.g., "My Project Key") so you can easily identify it later.

Copy and Save Your Key: Your new secret key will be displayed. This is the only time you will see the full key. Copy it immediately and save it in a secure location, like a password manager.[3][4]
Important: For the API key to be active, you may need to set up a payment method under the "Billing" section of your OpenAI account settings. The API is not free and operates on a pay-as-you-go basis.[1][5]

2. Set Up Your Environment File
Create a .env file: In the root directory of this project, create a new file and name it .env.

Add Your Key: Open the .env file with a text editor and add the following line, replacing "your_api_key_here" with the secret key you just copied from OpenAI:

OPENAI_API_KEY="your_api_key_here"

