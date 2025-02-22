import os
import requests
import re

# List of book IDs to download
book_ids = [25165,75316, 64223,43048,29542,37685,30689,24197,67345,67996]  #25 most popular fantasy books
corpus_folder = "corpus_text"
os.makedirs(corpus_folder, exist_ok=True)
book_filenames = []

def extract_main_content(text):
    match = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*(.*?)\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*', text, re.DOTALL)
    if not match:
        return None
    
    content = match.group(1).strip()

    content = re.sub(r'Produced by .*?\n', '', content, flags=re.IGNORECASE)  # Remove producer credits
    content = re.sub(r'\[Illustration:.*?\]', '', content, flags=re.DOTALL)  # Remove illustration captions
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Remove excessive blank lines
    content = re.sub(r'(_{3,}|-{3,}|\*{3,})', '', content)  # Remove decorative separators
    content = re.sub(r'[“”]', '"', content)  # Standardize quotation marks
    content = re.sub(r"[’]", "'", content)  # Standardize apostrophes

    content = re.sub(r'(?i)chapter\s+\d+', '', content)  # Remove "Chapter X"
    content = re.sub(r'\bPage\s+\d+\b', '', content)  # Remove "Page X"

    # Remove any extra spaces at start/end of lines
    content = "\n".join(line.strip() for line in content.splitlines())

    # Convert to lowercase (optional, useful for NLP tasks)
    content = content.lower()
    
    return content.strip()

# Download books
for book_id in book_ids:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    filename = f"{book_id}.txt"

    raw_text = response.text

    cleaned_text = extract_main_content(raw_text)

    if cleaned_text:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(cleaned_text)
        book_filenames.append(filename)
        print(f"Downloaded and cleaned {filename}")
    else:
        print(f"Skipping {filename} (content not found)")

# Combine all downloaded books into a single file
combined_text = "\n\n".join(open(fn, "r", encoding="utf-8").read() for fn in book_filenames)
# Save the combined text
combined_filename = "combined_books.txt"
with open(combined_filename, "w", encoding="utf-8") as f:
    for filename in book_filenames:
        with open(filename, "r", encoding="utf-8") as book_file:
            f.write(book_file.read() + "\n\n")

print(f"Dataset prepared: {combined_filename}")