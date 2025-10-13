import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def return_book(member_id, book_id):
    record = sb.table("borrow_records").select("*").eq("member_id", member_id).eq("book_id", book_id).is_("return_date", None).execute().data
    if not record:
        return {"error": "No active borrow record"}
    sb.table("borrow_records").update({"return_date": datetime.now().isoformat()}).eq("record_id", record[0]["record_id"]).execute()
    book = sb.table("books").select("stock").eq("book_id", book_id).execute().data[0]
    sb.table("books").update({"stock": book["stock"] + 1}).eq("book_id", book_id).execute()
    return {"success": True}

if __name__ == "__main__":
    mid = int(input("Member ID: "))
    bid = int(input("Book ID: "))
    print(return_book(mid, bid))
