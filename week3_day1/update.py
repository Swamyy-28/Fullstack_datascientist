import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def update_book_stock(book_id, new_stock):
    return sb.table("books").update({"stock": new_stock}).eq("book_id", book_id).execute().data

def update_member_email(member_id, new_email):
    return sb.table("members").update({"email": new_email}).eq("member_id", member_id).execute().data

if __name__ == "__main__":
    bid = int(input("Enter book_id to update stock: "))
    stock = int(input("Enter new stock: "))
    print(update_book_stock(bid, stock))

    mid = int(input("\nEnter member_id to update email: "))
    email = input("Enter new email: ")
    print(update_member_email(mid, email))
