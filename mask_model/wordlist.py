import csv

def split_data(file_path):
    result = []
    
    # Open the CSV file and read it using the csv.reader
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        
        # Skip the header row
        next(reader)
        
        # Process each row
        for row in reader:
            if len(row) == 5:  # Ensure the row has all 5 expected elements
                tweet_id, keyword, location, text, label = row
                # Append the tuple (with label converted to int)
                if tweet_id != "id":
                    result.append((int(tweet_id), keyword.lower().strip(), location, text.lower().strip(), int(label)))
            if len(row) == 4:
                tweet_id, keyword, location, text = row
                # Append the tuple (with label converted to int)
                if tweet_id != "id":
                    result.append((int(tweet_id), keyword.lower().strip(), location, text.lower().strip()))
    print(f"Result length is: {len(result)}")
    return result


def getwords(processed_data):
    wordlist = []

    for data in processed_data:
        if (data[1] != '' and data[1] not in wordlist):
            wordlist.append(data[1].lower().strip())

    return wordlist

vals = split_data("nlp-getting-started/test.csv")
