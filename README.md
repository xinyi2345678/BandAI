**Getting Started
Please access our streamlit app at [https://bandai-bv3vvjdpz8pdpic5ziqxnz.streamlit.app/]([url](https://bandai-bv3vvjdpz8pdpic5ziqxnz.streamlit.app/))

To run locally with your own AI,
1. Unzip BandAI.zip to extract all relevant files.

2. pip install -r requirements.txt

3. streamlit run BandAI.py

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

3. Set Up Google Sheets API Access (to update the google sheets embedded in Looker Studio Dashboard)

To enable the app to read/write results to Google Sheets, you must create your own Google Cloud Service Account key.

### Steps:
a. **Go to Google Cloud Console**  
   Visit [https://console.cloud.google.com/](https://console.cloud.google.com/) and log in with your Google account.
b. **Create a New Project (optional)**  
   - In the top navigation bar, click the project dropdown.  
   - Click **New Project** and give it a name (e.g., `BandAI`).  
   - Select or create a billing account if prompted.
c. **Enable APIs**  
   - In the left sidebar, go to **APIs & Services → Library**.  
   - Search for and enable:  
     - **Google Sheets API**  

d. **Create a Service Account**  
   - In the left sidebar, go to **IAM & Admin → Service Accounts**.  
   - Click **+ Create Service Account**.  
   - Give it a name (e.g., `bandai-sheets`).  
   - Assign role: **Editor** (this allows read/write access to Sheets and Drive).  
   - Finish and save.

e. **Generate a Service Account Key**  
   - Click your newly created service account.  
   - Go to **Keys → Add Key → Create New Key**.  
   - Choose **JSON** format and download the file.  
   - This JSON file contains your API credentials.

f. **Share Your Target Google Sheet**  
   - Open the Google Sheet [https://docs.google.com/spreadsheets/d/1sHmOYgL-J6AsYRg-4Bhap9ARxRfWMUEYyT3CEVT98i4/edit?gid=0#gid=0](https://docs.google.com/spreadsheets/d/1sHmOYgL-J6AsYRg-4Bhap9ARxRfWMUEYyT3CEVT98i4/edit?usp=sharing) that's embedded behind the Dashboard.  
   - Click **Share** and add the **Service Account email** (something like `bandai-sheets@your-project-id.iam.gserviceaccount.com`).  
   - Give it **Editor** access.

g. **Add Key to Environment File**  
   - Open your `.env` file (create it in the project root if it doesn’t exist).  
   - Paste the entire JSON key as one line like this:

     ```
     GOOGLE_SERVICE_ACCOUNT_JSON='{
       "type": "service_account",
       "project_id": "your-project-id",
       "private_key_id": "xxxxxxxxxxxxxxx",
       "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE...==\n-----END PRIVATE KEY-----\n",
       "client_email": "bandai-sheets@your-project-id.iam.gserviceaccount.com",
       "client_id": "1234567890",
       ...
     }'
     ```
