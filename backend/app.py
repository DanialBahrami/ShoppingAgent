import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import psycopg2
from config import DATABASE_URL
from agent import create_agent_executor, process_query
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images
app.mount("/images", StaticFiles(directory="backend/static/images"), name="images")

# Database connection
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Initialize AI Agent
agent = create_agent_executor()

@app.get("/")
def root():
    """Root endpoint for API status."""
    return {"message": "Welcome to the AI Shopping Assistant API!"}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Fetch product details by ID."""
    cursor.execute("SELECT * FROM products WHERE id = %s", (product_id,))
    product = cursor.fetchone()

    if not product:
        return {"error": "Product not found"}

    # Construct image URL
    image_url = f"http://127.0.0.1:8000/images/{product[10]}"

    return {
        "id": product[0],
        "gender": product[1],
        "master_category": product[2],
        "sub_category": product[3],
        "article_type": product[4],
        "base_color": product[5],
        "season": product[6],
        "year": product[7],
        "usage": product[8],
        "product_display_name": product[9],
        "image_url": image_url,
    }

@app.get("/search/{query}")
def search_products(query: str):
    """AI-powered product search using ChromaDB vector stores."""
    rec = process_query(agent, query)
    # 'rec' is now a dict with {message: str, products: list}
    return {"recommendations": rec}
