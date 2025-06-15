# Using requests library for embeddings
import requests
import numpy as np

def generate_embeddings():
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI5ZTM4NDkxZi1mYWEyLTQ2NGItODRmZS1mMjkzNDQzOTI0MDQiLCJlbWFpbCI6ImRoaW5lc2hwYXphbmlzYW15QGdtYWlsLmNvbSIsImlhdCI6MTczNDcxMzU5MCwiZXhwIjoxNzY2MjQ5NTkwfQ.LbJOEGf_OHN_gbDwbq4vpN22Ft8tk61HPglbX-9vzYk"
    }
    payload = {
        "input": "The food was delicious and the service was excellent.",
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embedding = np.array(data['data'][0]['embedding'])
    
    print(f"Generated embedding with shape: {embedding.shape}")
    print(f" total embedding", embedding)
    print(f"First 5 values: {embedding[:5]}")
    
    # Example: Calculate vector norm
    norm = np.linalg.norm(embedding)
    print(f"Vector norm: {norm}")
    
    return embedding

# Generate embeddings for two sentences to compare them
# embedding1 = generate_embeddings()

# print(len(embedding1))


def generate_completion():
    url = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI5ZTM4NDkxZi1mYWEyLTQ2NGItODRmZS1mMjkzNDQzOTI0MDQiLCJlbWFpbCI6ImRoaW5lc2hwYXphbmlzYW15QGdtYWlsLmNvbSIsImlhdCI6MTczNDcxMzU5MCwiZXhwIjoxNzY2MjQ5NTkwfQ.LbJOEGf_OHN_gbDwbq4vpN22Ft8tk61HPglbX-9vzYk"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Write a poem about artificial intelligence"
            }
        ],
        "model": "gpt-4.1-nano",
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    print(data)

#generate_completion()

def chunk_text(text, max_words=100):
    words = text.split()
    #print(words)
    chunks= []

    #print(len(words))
    for word in range(0, len(words), max_words):
        #(word)
        #print(word+max_words)
        chunks.append(" ".join(words[word:word+max_words]))
    return chunks

with open("founder_story.txt", "r", encoding='utf-8') as f:
    raw_text = f.read()


chunks = chunk_text(raw_text)
#print(chunks)
print(f"total chunks: {len(chunks)}")
print(chunks[0]) #there will be total 100 words in each chunk

