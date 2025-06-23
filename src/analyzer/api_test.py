import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
apikey = os.getenv("API_KEY")
url = os.getenv("URL")
project_id = os.getenv("PROJECT_ID")

# Validate that all required environment variables are loaded
if not all([apikey, url, project_id]):
    raise ValueError(
        "Missing required environment variables. Please check your .env file.")

# Define generation parameters
generate_params = {
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0.5,
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE
}

# Initialize the model with a supported model ID
model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",  # Updated to supported model
    credentials={"apikey": apikey, "url": url},
    params=generate_params,
    project_id=project_id
)

# Generate output
response = model.generate(
    prompt="What is 5 + 3?",
)

print(response['results'][0]['generated_text'])
