import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def borrow_book(member_id, book_id):
    book = sb.table("books").select("stock").eq("book_id", book_id).single().execute().data
    if not book:
        return "Book not found"
    if book["stock"] <= 0:
        return "Book not available"
    sb.table("books").update({"stock": book["stock"] - 1}).eq("book_id", book_id).execute()
    sb.table("borrow_records").insert({"member_id": member_id, "book_id": book_id}).execute()
    return "Book borrowed successfully"

if __name__ == "__main__":
    mid = int(input("Enter member_id: "))
    bid = int(input("Enter book_id: "))
    print(borrow_book(mid, bid))
