import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")
import psycopg2
import pandas as pd
from config import DATABASE_URL, DATA_PATH, IMAGE_DIR

def create_table():
    """Create a table for storing product data."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            gender TEXT,
            master_category TEXT,
            sub_category TEXT,
            article_type TEXT,
            base_color TEXT,
            season TEXT,
            year INT,
            usage TEXT,
            product_display_name TEXT,
            image_filename TEXT
        );
    """)
    conn.commit()
    conn.close()

def load_data():
    """Load data from styles.csv into the PostgreSQL database."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Read styles.csv
    data = pd.read_csv(DATA_PATH, on_bad_lines='skip')

    for _, row in data.iterrows():
        cursor.execute("""
            INSERT INTO products (id, gender, master_category, sub_category, article_type,
                                  base_color, season, year, usage, product_display_name, image_filename)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (row['id'], row['gender'], row['masterCategory'], row['subCategory'],
              row['articleType'], row['baseColour'], row['season'], int(row['year']) if not pd.isna(row['year']) else None,
              row['usage'], row['productDisplayName'], f"{row['id']}.jpg"))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table()
    load_data()
    print("Data loaded into the database successfully!")
