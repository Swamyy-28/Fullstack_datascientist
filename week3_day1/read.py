import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def list_books():
    return sb.table("books").select("*").execute().data

def search_books(field, value):
    return sb.table("books").select("*").ilike(field, f"%{value}%").execute().data

def member_with_books(member_id):
    member = sb.table("members").select("*").eq("member_id", member_id).execute().data
    borrowed = sb.table("borrow_records").select("*, books(title,author)").eq("member_id", member_id).execute().data
    return {"member": member, "borrowed_books": borrowed}

if __name__ == "__main__":
    print("All Books:")
    for b in list_books():
        print(b)

    f = input("\nSearch by (title/author/category): ")
    v = input("Enter search text: ")
    for b in search_books(f, v):
        print(b)

    mid = int(input("\nEnter member_id to view details: "))
    print(member_with_books(mid))
