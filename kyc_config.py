from dotenv import load_dotenv
import os

load_dotenv()

azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
json_loc = 'ocr/'
output_loc = 'out/'