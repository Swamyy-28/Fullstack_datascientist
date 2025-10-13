import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def list_products():
    resp = sb.table("products").select("*").execute()
    return resp.data

if __name__ == "__main__":
    products = list_products()
    for p in products:
        print("-----")
        for key, value in p.items():
            print(f"{key.capitalize()}: {value}")