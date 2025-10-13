import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
sb: Client = create_client(url, key)

def delete_product(sku):
    resp = sb.table("products").delete().eq("sku", sku).execute()
    return resp.data

if __name__ == "__main__":
    sku = input("Enter SKU to delete: ").strip()
    deleted = delete_product(sku)
    if deleted:
        print("Deleted:", deleted)
    else:
        print("No product found with that SKU.")