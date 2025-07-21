import re

def count_tokens(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()  # Convert to lowercase

    tokens = re.findall(r'\b\w+\b', text)  # Extract words as tokens
    total_tokens = len(tokens)  # Count total tokens

    return total_tokens

# Example usage
file_path = "file.txt"  # Change this to your file path
total_tokens = count_tokens(file_path)

print(f"Total token count: {total_tokens}")
