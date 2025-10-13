import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def add_member(name, email):
    return sb.table("members").insert({"name": name, "email": email}).execute().data

def add_book(title, author, category, stock):
    return sb.table("books").insert({
        "title": title, "author": author, "category": category, "stock": stock
    }).execute().data

if __name__ == "__main__":
    # First add a member
    name = input("Enter member name: ")
    email = input("Enter member email: ")
    print("Inserted Member:", add_member(name, email))

    # Then add a book
    title = input("Enter book title: ")
    author = input("Enter book author: ")
    category = input("Enter category: ")
    stock = int(input("Enter stock: "))
    print("Inserted Book:", add_book(title, author, category, stock))
