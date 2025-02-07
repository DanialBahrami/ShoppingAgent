from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Database connection URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Other configs
DATA_PATH = "data_small/styles.csv"
IMAGES_FOLDER = "data_small/images"
VECTORSTORE_TEXT_PATH = "faiss_text_store"
VECTORSTORE_IMAGE_PATH = "faiss_image_store"

# Image Path for database
IMAGE_DIR = "backend/static/images/"