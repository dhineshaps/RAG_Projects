import requests
import numpy as np
import faiss

EURI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI5ZTM4NDkxZi1mYWEyLTQ2NGItODRmZS1mMjkzNDQzOTI0MDQiLCJlbWFpbCI6ImRoaW5lc2hwYXphbmlzYW15QGdtYWlsLmNvbSIsImlhdCI6MTczNDcxMzU5MCwiZXhwIjoxNzY2MjQ5NTkwfQ.LbJOEGf_OHN_gbDwbq4vpN22Ft8tk61HPglbX-9vzYk"
EURI_API_URL = "https://api.euron.one/api/v1/euri/alpha"


def generate_embeddings(text, model="text-embedding-3-small"):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": model
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embedding = np.array(data['data'][0]['embedding'])
    
    # print(f"Generated embedding with shape: {embedding.shape}")
    # print(f" total embedding", embedding)
    # print(f"First 5 values: {embedding[:5]}")
    
    # Example: Calculate vector norm
    norm = np.linalg.norm(embedding)
    # print(f"Vector norm: {norm}")
    
    return embedding

#########################################################################################
def generate_completion(prompt, model="gpt-4.1-nano"):
    url = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI5ZTM4NDkxZi1mYWEyLTQ2NGItODRmZS1mMjkzNDQzOTI0MDQiLCJlbWFpbCI6ImRoaW5lc2hwYXphbmlzYW15QGdtYWlsLmNvbSIsImlhdCI6MTczNDcxMzU5MCwiZXhwIjoxNzY2MjQ5NTkwfQ.LbJOEGf_OHN_gbDwbq4vpN22Ft8tk61HPglbX-9vzYk"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens":  500,
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'].strip()


###############################################################################
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
#print(type(chunks))
#print(f"total chunks: {len(chunks)}")
#print(chunks[0]) #there will be total 100 words in each chunk

embedding1 = generate_embeddings(chunks[0]) #all will have 1536 lenth only, we can also hardcode it

########################################################################################

vec_len = embedding1.shape[0]

index = faiss.IndexFlatL2(vec_len)  # Create a flat index for L2 distance

chunk_mapping = []

for chunk in chunks:
    embedding = generate_embeddings(chunk)
    # print(np.array([embedding]))
    index.add(np.array([embedding]).astype(np.float32))  # Add the embedding to the index as faiss only accepts float32 #converts embedding to 2d array
    chunk_mapping.append(chunk)  # Keep track of which chunk corresponds to which 
    

# print(chunk_mapping)
#faiss.write_index(index, "rag_index.faiss")  # Save the index to a file

#################################################################################################

def retrieve_top_k_chunks(query, k=3):
    query_embedding = generate_embeddings(query)
    query_embedding = np.array([query_embedding]).astype(np.float32)  # Convert to 2D array for faiss
    distances, indices = index.search(query_embedding, k)  # Search the index for the top k nearest neighbors
    
    results = []
    for i in range(k):
        chunk_index = indices[0][i]
        distance = distances[0][i]
        results.append((chunk_mapping[chunk_index], distance))
    
    return results

#print(results)

def build_prompt(context_chunks, query):
    context = "\n\n".join([doc for doc, _ in context_chunks])
    return f"""use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {query}
    Answer:"""      

query = "did he started inueron"
op = retrieve_top_k_chunks(query)

prompt = build_prompt(op,query)

print(generate_completion(prompt))
