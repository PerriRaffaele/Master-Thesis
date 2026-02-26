from datasets import load_dataset
import random

def build_background_dataset(num_samples=500):
    background_texts = []
    
    # 1. Fetch Plain English (WikiText)
    print("Downloading English background data...")
    wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    english_samples = 0
    for row in wiki:
        text = row["text"].strip()
        # Only take substantial paragraphs, not short titles
        if len(text) > 200: 
            background_texts.append(text)
            english_samples += 1
        if english_samples >= (num_samples // 2):
            break

    # 2. Fetch Boilerplate/Non-Algorithmic Python (The Stack Smol)
    print("Downloading Python background data...")
    # The Stack Smol is a tiny, manageable subset of GitHub code
    stack = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    
    python_samples = 0
    # Keywords that usually indicate boilerplate/web/config code, NOT algorithms
    boring_keywords = ["django", "flask", "setup(", "argparse", "sqlalchemy", "route(", "html"]
    
    for row in stack:
        code = row["content"]
        # Only keep the code if it contains one of our "boring" keywords
        if any(kw in code.lower() for kw in boring_keywords) and len(code) > 100:
            # Truncate to roughly the length of a benchmark prompt to keep things balanced
            background_texts.append(code[:1000]) 
            python_samples += 1
        if python_samples >= (num_samples // 2):
            break

    # Shuffle the dataset so the English and Python are mixed randomly
    random.shuffle(background_texts)
    print(f"Built background dataset with {len(background_texts)} samples!")
    
    return background_texts

# Usage:
# target_texts = [row['prompt'] + row['canonical_solution'] for _, row in benchmark_df.iterrows()]
# background_texts = build_background_dataset(num_samples=len(target_texts))