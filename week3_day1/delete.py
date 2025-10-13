import os
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def delete_member(member_id):
    borrowed = sb.table("borrow_records").select("*").eq("member_id", member_id).is_("return_date", None).execute().data
    if borrowed:
        return "Cannot delete: member has borrowed books"
    return sb.table("members").delete().eq("member_id", member_id).execute().data

def delete_book(book_id):
    borrowed = sb.table("borrow_records").select("*").eq("book_id", book_id).is_("return_date", None).execute().data
    if borrowed:
        return "Cannot delete: book is borrowed"
    return sb.table("books").delete().eq("book_id", book_id).execute().data
if __name__ == "__main__":
    mid = int(input("Enter member_id to delete: "))
    print(delete_member(mid))

    bid = int(input("\nEnter book_id to delete: "))
    print(delete_book(bid))
