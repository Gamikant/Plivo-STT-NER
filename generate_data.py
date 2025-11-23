import json
import random
import os

# --- Configuration ---
NUM_TRAIN = 1000
NUM_DEV = 200
OUTPUT_DIR = "data"

# --- Vocabulary & Templates ---
FIRST_NAMES = ["ramesh", "suresh", "priyanka", "rohan", "aditi", "vikram", "sneha", "rahul", "amit", "deepak", "anjali", "neha", "kavita", "arjun", "varun"]
LAST_NAMES = ["sharma", "verma", "gupta", "mehta", "singh", "patel", "kumar", "yadav", "reddy", "nair", "iyer", "malhotra"]
CITIES = ["mumbai", "delhi", "bangalore", "chennai", "hyderabad", "pune", "kolkata", "ahmedabad", "jaipur", "lucknow"]
LOCATIONS = ["mg road", "indira nagar", "connaught place", "juhu beach", "whitefield", "cyber city", "marine drive", "worli", "koramangala"]
DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "example"]
TLDS = ["com", "in", "co dot in", "org", "net"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

# Number words for noise injection
DIGIT_MAP = {
    "0": ["zero", "oh", "naught"],
    "1": ["one"],
    "2": ["two"],
    "3": ["three"],
    "4": ["four"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight"],
    "9": ["nine"]
}

def get_random_digit_str(length=10):
    return "".join([str(random.randint(0, 9)) for _ in range(length)])

def noise_number(num_str):
    """Convert digits to spoken words randomly."""
    # 30% chance to speak out digits, 70% keep as digits with spaces
    if random.random() < 0.3:
        words = []
        for char in num_str:
            words.append(random.choice(DIGIT_MAP[char]))
        return " ".join(words)
    else:
        # Add random spacing: "987 65" or "98765"
        if random.random() < 0.5:
            return " ".join(num_str) # Space out every digit
        return num_str

# --- Generators ---

def gen_email():
    """Generates a noisy email."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    # Noisy format: "john dot doe at gmail dot com"
    email_text = name.replace(" ", " dot ") + " at " + random.choice(DOMAINS) + " dot " + random.choice(TLDS)
    return email_text.lower(), "EMAIL"

def gen_phone():
    """Generates a phone number (digits or spoken)."""
    raw_num = get_random_digit_str(10)
    return noise_number(raw_num), "PHONE"

def gen_card():
    """Generates a credit card number."""
    # 16 digits usually
    raw_num = get_random_digit_str(16)
    # Often grouped in 4s: "4242 4242..."
    groups = [raw_num[i:i+4] for i in range(0, 16, 4)]
    text = " ".join(groups)
    # Apply noise
    if random.random() < 0.2:
        text = noise_number(raw_num) # Fully spoken
    return text, "CREDIT_CARD"

def gen_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}", "PERSON_NAME"

def gen_date():
    day = random.randint(1, 31)
    year = random.randint(2020, 2030)
    # Formats: "15 august 2024", "15 08 2024", "august 15"
    r = random.random()
    if r < 0.33:
        return f"{day} {random.choice(MONTHS)} {year}", "DATE"
    elif r < 0.66:
        return f"{day:02d} {random.randint(1,12):02d} {year}", "DATE"
    else:
        return f"{random.choice(MONTHS)} {day}", "DATE"

def gen_city():
    return random.choice(CITIES), "CITY"

def gen_location():
    return random.choice(LOCATIONS), "LOCATION"

# --- Sentence Templates ---
TEMPLATES = [
    "my number is {PHONE}",
    "contact me on {PHONE} immediately",
    "call {PHONE}",
    "reach out to {EMAIL}",
    "my email id is {EMAIL}",
    "send the details to {EMAIL} please",
    "i live in {CITY}",
    "i am travelling to {CITY} and then {LOCATION}",
    "meet me at {LOCATION}",
    "my card number is {CREDIT_CARD}",
    "pay using {CREDIT_CARD}",
    "use card {CREDIT_CARD} for the transaction",
    "my name is {PERSON_NAME}",
    "this is {PERSON_NAME} speaking",
    "i was born on {DATE}",
    "schedule the meeting for {DATE}",
    "from {CITY} to {CITY} on {DATE}",
    # Complex multi-entity
    "my name is {PERSON_NAME} and email is {EMAIL}",
    "contact {PERSON_NAME} at {PHONE}",
    "{PERSON_NAME} lives in {CITY}",
    "payment by {CREDIT_CARD} confirmed for {DATE}"
]

def generate_sample(uid_prefix, idx):
    template = random.choice(TEMPLATES)
    text_parts = []
    entities = []
    
    # We split template by spaces to handle reconstruction carefully? 
    # Actually simpler: Replace placeholders one by one and track indices.
    
    # Identify what needs replacing
    placeholders = ["{PHONE}", "{EMAIL}", "{CITY}", "{LOCATION}", "{CREDIT_CARD}", "{PERSON_NAME}", "{DATE}"]
    
    # To handle offsets, we build the string part by part
    # This is a bit tricky with simple string replace. Let's do a token-based build.
    
    # Parse template into chunks
    # e.g. "my name is {PERSON_NAME} and email is {EMAIL}" -> ["my name is ", "{PERSON_NAME}", " and email is ", "{EMAIL}"]
    
    parts = []
    last_pos = 0
    # Find all placeholders in order
    found_placeholders = []
    for ph in placeholders:
        if ph in template:
            found_placeholders.append((template.find(ph), ph))
    
    # If multiple of same type? This simple logic assumes distinct placeholders or basic ones. 
    # For robust randomness, let's just pick a generator based on the key found.
    
    # Better approach: Split by space and reconstruct
    tokens = template.split()
    final_text = ""
    
    for token in tokens:
        entity_val = None
        entity_label = None
        
        # Check if token is a placeholder key
        clean_token = token.replace(",", "").replace(".", "") # simple cleanup
        
        if "{PHONE}" in token:
            entity_val, entity_label = gen_phone()
        elif "{EMAIL}" in token:
            entity_val, entity_label = gen_email()
        elif "{CREDIT_CARD}" in token:
            entity_val, entity_label = gen_card()
        elif "{PERSON_NAME}" in token:
            entity_val, entity_label = gen_name()
        elif "{DATE}" in token:
            entity_val, entity_label = gen_date()
        elif "{CITY}" in token:
            entity_val, entity_label = gen_city()
        elif "{LOCATION}" in token:
            entity_val, entity_label = gen_location()
            
        if entity_val:
            start = len(final_text)
            if final_text: # Add space if not start
                final_text += " "
                start += 1
            
            final_text += entity_val
            end = len(final_text)
            entities.append({
                "start": start,
                "end": end,
                "label": entity_label
            })
        else:
            if final_text:
                final_text += " "
            final_text += token
            
    return {
        "id": f"{uid_prefix}_{idx:04d}",
        "text": final_text,
        "entities": entities
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate Train
    with open(f"{OUTPUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        for i in range(NUM_TRAIN):
            sample = generate_sample("train", i)
            f.write(json.dumps(sample) + "\n")
    
    # Generate Dev
    with open(f"{OUTPUT_DIR}/dev.jsonl", "w", encoding="utf-8") as f:
        for i in range(NUM_DEV):
            sample = generate_sample("dev", i)
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {NUM_TRAIN} training samples and {NUM_DEV} dev samples in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()