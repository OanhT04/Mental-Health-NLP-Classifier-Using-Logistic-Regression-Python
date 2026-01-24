
import pandas as pd
import contractions
import re
import wordninja

#Oanh Tran
#W!!!!!  using all of these functions may be excessive/overkill and could remove too much data adjust as needed


# -------- Slang normalization --------
SLANG_MAP = {
    "idk": "i do not know",
    "imo": "in my opinion",
    "im": "i am",
    "ive": "i have",
    "dont": "do not",
    "cant": "can not",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "bc": "because",
    "tho": "though",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "rn": "right now",
    "omg": "oh my god",
    "wtf": "what the fuck",
    "kys": "kill yourself",
    "kms": "kill myself",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "rn": "right now",
}

def is_scrambled(text):
    words = text.split()
    if not words:
        return True

    vowel_words = sum(1 for w in words if re.search(r"[aeiou]", w))
    if vowel_words / len(words) < 0.5:
        return True

    if re.search(r"(.)\1{4,}", text):
        return True

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 15:
        return True

    return False

def fix_text(text):
    return re.sub(r"(.)\1{2,}", r"\1", str(text))


# -------- Add missing spaces --------
def add_missing_spaces(text):
    if not isinstance(text, str):
        return text
    return " ".join(wordninja.split(text))



def remove_duplicated_words(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Case 1: glued duplicates like "hellohello"
    text = re.sub(r"\b([a-zA-Z]+)\1+\b", r"\1", text)
    # Case 2: repeated words like "hello hello hello"
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    return text

def remove_words_no_vowels(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"\b[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]+\b", "", text)

def normalize_slang(text):
    return " ".join(SLANG_MAP.get(w.lower(), w) for w in text.split())




def main():
    #change to whatever file path you need
    input_file = "mental_health_combined_test.csv"
    output_file = "mental_unbalanced_cleaned.csv"

    # -------- Load data --------
    df = pd.read_csv(input_file)
    print("Data loaded successfully.")
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    # -------- Standardize labels --------
    df["status"] = df["status"].str.strip().str.title()

    df["status"] = df["status"].replace({
        "Suicidal": "High_Risk_Emotional_Distress",
        "Stress": "Stress",
        "Anxiety": "Anxiety",
        "Depression": "Depression",
        "Normal": "No_Distress",
    })

    # -------- Drop missing values--------
    df = df.dropna(subset=["text"]).copy()

    # -------- Normalize text --------
    df["text"] = df["text"].astype(str).str.lower()
    df["text"] = df["text"].apply(contractions.fix)
    df["text"] = df["text"].apply(normalize_slang)
    df["text"] = df["text"].astype(str).str.replace(",", "", regex=False)

    # -------- Regex cleaning --------
    df["text"] = df["text"].str.replace(r"http\S+|www\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"[^a-z\s]", "", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["text"] = df["text"].apply(add_missing_spaces)
    # -------- Remove short entries --------
    df = df[df["text"].str.split().str.len() >= 2].copy()
    df["text"] = df["text"].apply(remove_duplicated_words)
    df["text"] = df["text"].apply(remove_words_no_vowels)
    # -------- Scrambled text removal --------
    df = df[~df["text"].apply(is_scrambled)].copy()
    
    # -------- Remove duplicates --------
    df = df.drop_duplicates(subset=["text"])
    
    #-------- fix some longgggggg words---------
    df["text"] = df["text"].apply(fix_text)

    # -------- Save --------
    df.to_csv(output_file, index=False)
    print(f"Data cleaned successfully. Saved to: {output_file}")


if __name__ == "__main__":
    main()

