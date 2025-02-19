import os

with open("The Wealth of Nations.txt", "r", encoding="utf-8") as f1, open("A Critique of Political Economies.txt", "r", encoding="utf-8") as f2:
    text = f1.read() + "\n" + f2.read()

with open("combined_books.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Dataset prepared: combined_books.txt")
