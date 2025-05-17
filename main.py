import os
import faiss
import pickle
import openai
import numpy as np
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import datetime
import pickle
import tiktoken
import json
from typing import Optional
import json
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# openai.api_key = os.getenv("API_KEY")

client = OpenAI(
    api_key=os.getenv("API_KEY")
)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


app = FastAPI()

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


today = datetime.datetime.now()
day = datetime.datetime.now().strftime('%A')
tokens = 0

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens




# This is the code for the chatbot
#*********************************
def get_completion_from_messages(messages, model='gpt-4o-mini', temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    response_bot = response.choices[0].message.content
    response_bot = response_bot
    print('********************************',response_bot)
    return response_bot

class User:
    user_dict = {}
    id = ''
    context = []
    
    status = ''

    def __init__(self, id, status):
        self.id = id
        self.user_dict = {}
        self.status = status

    def create_user(self):
        user_ids = self.user_dict.values()
        status = self.status
        if self.id not in user_ids:
            if status == 'cost-estimator':
                self.context = [
                    {'role': 'system', 'content': f"""You are an AI chatbot designed to assist users with service cost estimations and act as a general-purpose AI assistant with reasoning-based responses. Your goal is to interact like a human assistant, ask only one question at a time, collect necessary information, and then calculate costs dynamically or retrieve fixed costs from a database.
                     
                    ## **Multilingual Capability**
                    - Automatically detect the user's language from their message.
                    - Respond in the same language without requiring the user to specify it.
                    - Maintain natural fluency and coherence in responses across different languages.


                    Instructions:
                    1. Understanding User Requests:
                    If the user asks for a service cost estimation (e.g., app development, website creation, ad campaign, etc.), check if it falls under a predefined category with a fixed cost.
                    If the cost is fixed, provide that cost directly.
                    If the cost is not fixed, dynamically calculate it based on collected requirements.
                    If the request is general and not related to cost estimation, function as a reasoning-based general chatbot and provide helpful responses.
                    2. Collecting Requirements:
                    Always ask only one question at a time to guide the user through a structured conversation.
                    Ensure you gather all necessary details before providing a final cost estimation.
                    Example questions based on the service type:
                    App Development: Platform (Android, iOS, Web), Features (Payment Integration, Admin Panel, etc.), Complexity Level
                    Ad Campaign: Platform (Google Ads, Facebook Ads), Budget, Target Audience, Duration
                    3. Cost Calculation:
                    If the cost for a service is predefined in a database, retrieve and display that cost.
                    If no fixed cost exists, analyze the gathered requirements and estimate a dynamic cost based on logical reasoning and industry standards.
                    Clearly break down the cost into components for transparency.
                    4. Handling Edge Cases:
                    If a user provides insufficient details, politely ask for clarification before giving a price.
                    If the request is out of scope, provide a general response or guide the user to a relevant alternative.
                    Example Scenarios:
                    Example 1: App Development (Fixed Cost)
                    User: "I want to develop an e-commerce app."
                    AI: "Great! Do you need it for Android, iOS, or both?"
                    User: "Both."
                    AI: "Which features do you need? (Payment integration, Admin Panel, User Authentication, etc.)"
                    User: "Payment integration and Admin Panel."
                    AI: "The fixed cost for an e-commerce app with these features on both Android and iOS is $12,000. Would you like a detailed breakdown?"

                    Example 2: Ad Campaign (Dynamic Cost Calculation)
                    User: "I want to run an ad campaign."
                    AI: "Sure! Which platform do you want to use? (Google Ads, Facebook Ads, Instagram Ads)"
                    User: "Google Ads."
                    AI: "What is your budget?"
                    User: "$500."
                    AI: "How long do you want to run the campaign? (1 week, 2 weeks, 1 month)"
                    User: "2 weeks."
                    AI: "Based on your budget and duration, the estimated cost is $250 for setup and $500 for ad spend. Would you like to proceed?"

                    Example 3: General Chatbot Interaction
                    User: "Who won the last FIFA World Cup?"
                    AI: "The last FIFA World Cup was won by [Team Name] in [Year]. Would you like more details on the final match?"
                    
                    ### *Example 4: Consulting Services*
                    *User:* "I need marketing consulting for my startup."  
                    *AI:* "Great! What specific areas do you need help with? (SEO, Social Media, Paid Ads, Branding)"  
                    *User:* "SEO and Social Media."  
                    *AI:* "How many consulting sessions are you looking for? (One-time, Weekly, Monthly)"  
                    *User:* "Weekly for two months."  
                    *AI:* "Based on your request, the estimated cost for weekly consulting over two months is $3,000. Would you like to proceed?"  

                    ---

                    ## *📌 Final Notes*
                    ✅ *Auto-detects & responds in the user's language*  
                    ✅ *Handles fixed & dynamic cost estimations*  
                    ✅ *Works as a general AI assistant*  
                    ✅ *Breaks down costs logically for better transparency* 


"""},
                    {'role': 'user', 'content': f''},
                    
                    ]
            
            

            self.user_dict[self.id] = self.context
            return "User created successfully!"
        
        
        else:
            return "User ID already exists!"

    def get_user_dict(self):
        return self.user_dict

    def get_user_context(self):
        if self.id in self.user_dict.keys():
            return self.user_dict[self.id]
        else:
            return 'User not found!'

    def save_context(self, context):
        path = './' + self.id + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(context, fp)
        return path

    def load_saved_dict(self):
        if os.path.exists(self.id + '.pkl'):
            with open(self.id + '.pkl', 'rb') as fp:
                dict = pickle.load(fp)
                self.user_dict = dict
                return dict
        else:
            return 'User not found!'



    def is_user(self):
        if os.path.exists('./' + self.id + '.pkl'):
            return self.load_saved_dict()
        else:
            return 0

    def delete_previous_context(self):
        context = self.context
        return context



# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/chat_endpoint")
async def chat_with_bot(user_id: str, message: str,):
    id = user_id
    status = "cost-estimator" 
    message = message
    tokens = 0
    client = User(id, status)
    is_user = client.is_user()
    if is_user == 0:
        client.create_user()
        context = client.get_user_context()
        context.append({'role': 'user', 'content': f"{message}"})
        print(context)

        message = message
        # token_prompt = word_tokenize(message)
        for i in context:
            for j in i.values():
                count = num_tokens_from_string(j,'cl100k_base')
                tokens += count
        if tokens >= 15500:
            context = client.delete_previous_context()
            print('i\'m a little tired, how about we continue our conversation later!')
        message = message

        response = get_completion_from_messages(context)
        
        context.append({'role': 'assistant', 'content': f"{response}"})
        print(context)
        path = client.save_context(context)
        return response, path

    else:
        context = client.load_saved_dict()
        context.append({'role': 'user', 'content': f"{message}"})

        message = message
        # token_prompt = word_tokenize(message)

        for i in context:
            for j in i.values():
                count = num_tokens_from_string(j,'cl100k_base')
                tokens += count
        if tokens >= 15500:
            context = client.delete_previous_context()
            print('i\'m a little tired, how about we continue our conversation later!')
        message = message
        
        response = get_completion_from_messages(context)

        context.append({'role': 'assistant', 'content': f"{response}"})
        print(context)
        path = client.save_context(context)

        return response
    
# Load or initialize user session history

def get_user_history(user_id: str):
    path = os.path.join(SESSIONS_DIR, f"{user_id}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


COUNTRY_SIMILARITY_THRESHOLDS = {
    "pakistan": 0.65,
    "canada": 0.63,
    "usa": 0.55,
    "uk": 0.52,
    # Add more countries as needed
}

def save_user_history(user_id: str, history):
    path = os.path.join(SESSIONS_DIR, f"{user_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(history, f)

# def extract_chunks_from_pdfs(country):
#     # Remove any leading or trailing spaces from the country name
#     country = country.strip()

#     folder_path = os.path.join("books", country)
#     if not os.path.exists(folder_path):
#         print(f"❌ Folder not found for country: {country}")
#         return [], []

#     chunks, metadata = [], []

#     for file in os.listdir(folder_path):
#         if file.endswith(".pdf"):
#             path = os.path.join(folder_path, file)
#             try:
#                 reader = PdfReader(path)
#                 text = ""
#                 for page in reader.pages:
#                     text += page.extract_text() or ""

#                 words = text.split()
#                 for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
#                     chunk = " ".join(words[i:i + CHUNK_SIZE])
#                     chunks.append(chunk)
#                     metadata.append({"book": file})

#             except Exception as e:
#                 print(f"⚠️ Error reading {file}: {e}")

#     return chunks, metadata


# def embed_texts(texts):
#     print("🔗 Creating embeddings...")
#     response = client.embeddings.create(input=texts, model=EMBED_MODEL)
#     return np.array([d.embedding for d in response.data])


# def search_faiss_index(query, index, chunks, metadata, top_k=3, threshold=1.0):
#     print("🔍 Searching for top chunks...")
#     query_embedding = client.embeddings.create(
#         input=[query],
#         model=EMBED_MODEL
#     ).data[0].embedding

#     query_embedding = np.array(query_embedding).reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k)

#     top_chunks = []
#     for dist, i in zip(distances[0], indices[0]):
#         if dist < threshold and i < len(chunks):
#             top_chunks.append((metadata[i]["book"], chunks[i]))
#     return top_chunks

# def query_gpt(user_query, history=None):
#     user_query = (
#         f"Answer the following question as accurately as possible:\n\n{user_query}\n\n"
#         f"If unsure, try to answer using your general knowledge."
#     )

#     print("🤖 Querying GPT-4o...")

#     messages = history if history else []
#     messages = messages + [{"role": "user", "content": user_query}]

#     response = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=messages,
#         temperature=0.3
#     )
#     return response.choices[0].message.content.strip()

def extract_chunks_from_pdfs(country):
    folder_path = os.path.join("books", country)
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found for country: {country}")
        return [], []

    chunks, metadata = [], []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = " ".join(words[i:i + CHUNK_SIZE])
                    chunks.append(chunk)
                    metadata.append({"book": file})

            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")

    return chunks, metadata



# def embed_texts(texts):
#     print("🔗 Creating embeddings...")
#     response = client.embeddings.create(input=texts, model=EMBED_MODEL)
#     return np.array([d.embedding for d in response.data])

def embed_texts(texts, batch_size=100):
    print("🔗 Creating embeddings...")
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBED_MODEL)
        embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)
# def search_faiss_index(query, index, chunks, metadata, top_k=3, threshold=1.0):
#     print("🔍 Searching for top chunks...")
#     query_embedding = client.embeddings.create(
#         input=[query],
#         model=EMBED_MODEL
#     ).data[0].embedding

#     query_embedding = np.array(query_embedding).reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k)

#     top_chunks = []
#     for dist, i in zip(distances[0], indices[0]):
#         if dist < threshold and i < len(chunks):
#             top_chunks.append((metadata[i]["book"], chunks[i]))
#     print(f"Top chunks:{top_chunks}")
#     return top_chunks

def search_faiss_index(query, index, chunks, metadata, top_k=3, min_similarity=0.61):
    print("🔍 Searching for top chunks (cosine similarity)...")

    query_embedding = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    ).data[0].embedding

    query_embedding = np.array(query_embedding).reshape(1, -1)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    distances, indices = index.search(query_embedding, top_k)
    print("📏 FAISS cosine similarities:", distances)
    print("📚 FAISS indices:", indices)

    print("📏 FAISS cosine similarities:", distances)
    print("📚 FAISS indices:", indices)

    top_chunks = []
    for sim, i in zip(distances[0], indices[0]):
        print(f"🔍 Match: sim={sim:.4f}, file={metadata[i]['book']}")
        if sim >= min_similarity and i < len(chunks):  # 🧠 Only use results with strong match
            top_chunks.append((metadata[i]["book"], chunks[i]))
            print(f"Similarity:{sim}")
            print(f"Top Chunks:{top_chunks}")

    print(f"✅ Found {len(top_chunks)} relevant chunks.")
    print(f"Top Chunks:{top_chunks}")
    return top_chunks


def query_gpt(user_query, history=None):
    user_query = (
        f"Answer the following question as accurately as possible:\n\n{user_query}\n\n"
        f"If unsure, try to answer using your general knowledge."
    )

    print("🤖 Querying GPT-4o...")

    messages = history if history else []
    messages = messages + [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def get_response_from_general_knowledge(user_query, country, history=None):
    
    system_prompt = f""""You are a dual-purpose intelligent chatbot. Your primary role is to act as a highly knowledgeable taxation assistant, capable of answering tax-related questions using country-specific rules, laws, and practices. 

        However, you are also capable of acting as a general AI assistant when the user asks a query unrelated to taxation.

        You must always detect whether a query is tax-related or general:
        - For **tax-related queries**, use your taxation expertise and follow the detailed rules below.
        - For **general queries**, respond as a helpful, reasoning-based assistant — give informative and logical answers without redirecting to tax unless relevant.

                🔹 You must respond according to the taxation rules, regulations, and legal practices of the {country} specified by the user. Always consider the local laws of the provided country when generating responses.

                🔹 If information is not country-specific or universal, clearly state that it's a general answer and may vary based on local laws.

                🔹 Always include explanations for why a question is being asked or why a specific rule or deduction is applied, to help the user understand the reasoning behind the guidance.

                If a user provides incomplete information, ask only one follow-up question at a time. Do not ask multiple follow-up questions in a single response. After the user answers, ask the next relevant question. Always explain why you are asking that question.

                Your primary role is to assist users with taxation. If the user asks a question unrelated to tax, respond like a general chatbot and provide reasoning based responses.

                📚 At the end of each response, include a "reference" field. If the information comes from a specific source, include the source name or title (if known). If the information is generated from your general knowledge, write:
                "reference": "General knowledge".

                🌐 Multilingual Capability:
                - Automatically detect the user's language from their message.
                - Respond in the same language without requiring the user to specify it.
                - Maintain natural fluency and coherence in responses across different languages.

                📋 Guidelines:

                Taxation Queries:
                - Categorize income sources (salary, business, rental, investment, etc.).
                - Apply correct tax slabs and deductions.
                - Explain the impact of different income sources on tax liability.
                - Highlight potential compliance issues or incorrect reporting risks.
                - Provide tax calculation examples when necessary.
                - Always provide explanations for any advice, rule, or calculation to help users understand the reasoning behind it.

                General Queries:
                - If the user's query is unrelated to taxation, switch roles and respond normally as a general chatbot.
                - Do not redirect the user back to taxation unless the user specifically asks to connect the topic to tax.
                - Answer questions with clear reasoning and helpful information, as a general-purpose assistant would.


                Clarifications & Follow-Ups(APPLY TO ALL KIND OF QUERIES):
                - If the user query lacks key details (e.g., year, income source, location), follow this rule:
                    • Ask only one follow-up question at a time.
                    • Do not list multiple questions in one message.
                    • Clearly explain why the question is being asked (e.g., "To determine applicable deductions, I need to know the tax year you're filing for.").
                    • Wait for the user’s reply before proceeding to the next question.
                - Use a step-by-step, interactive conversational approach with structured reasoning.

                Now, based on the user’s input and country, respond appropriately. For tax queries, guide them one step at a time using the specified country’s rules. For general queries, respond as a helpful assistant.

               **ALWAYS PROVIDE YOUR RESPONSE IN TGHE EXACT BELOW FORMAT DONOT USE ANY REGARDING THIS FORMAT**
                OUTPUT FORMAT:
                You must respond **only** with a valid JSON object in the following format and nothing else — no explanation, no additional text, no markdown:

                {{
                "response": "<your generated response>",
                "reference": "<your source>"
                }}

                ⚠️ Do not include any extra text before or after the JSON.
                ⚠️ Do not explain what the response is.
                ⚠️ Do not use code blocks or markdown formatting.
                ⚠️ Return the result strictly as raw JSON.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history
        
    
    messages.append({"role": "user", "content": user_query})

    # Add a final prompt to force the format
    messages.append({
        "role": "user",
        "content": 'Please respond strictly in the required JSON format as specified.'
    })


    print("🤖 Querying GPT-4o with general knowledge fallback prompt...")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0
    )
    res = response.choices[0].message.content
    return res


def get_response_from_books(user_query, country, history=None):
    print(f"\n🌍 Handling query for country: {country}")
    os.makedirs("faiss_indexes", exist_ok=True)

    index_path = f"faiss_indexes/{country}_index.faiss"
    metadata_path = f"faiss_indexes/{country}_metadata.pkl"

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("📦 Loading cached index and metadata...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            chunks, metadata = pickle.load(f)
    else:
        print("📄 Index not found. Generating...")
        chunks, metadata = extract_chunks_from_pdfs(country)
        print(f"📄 Total chunks loaded for {country}: {len(chunks)}")
        print("📚 Books included:", list(set(meta['book'] for meta in metadata)))
        if not chunks:
            
            print("⚠️ No relevant chunks found. Falling back to general knowledge.")
            fallback_response_raw = get_response_from_general_knowledge(user_query, country, history)
            print("get_response_from_general_knowledge")
            print(f"user query :{user_query}")
            print(f"country:{country}")
            print(f"History:{history}")
            print(f"DEBUG: fallback_response_raw = {repr(fallback_response_raw)}")
            print(fallback_response_raw)
            print(type(fallback_response_raw)) 
            if isinstance(fallback_response_raw, dict):
                fallback_response = fallback_response_raw
            else:
                try:
                    fallback_response = json.loads(fallback_response_raw)
                except (json.JSONDecodeError, TypeError):
                    # It's just a plain string (like what you got), not a JSON object
                    return {
                        "response": fallback_response_raw,
                        "reference": "N/A"
                    }

            # Extract response fields from JSON
            response = fallback_response.get("response", "No response found")
            reference = fallback_response.get("reference", "No reference found")

            return {
                "response": response,
                "reference": reference
            }
            # fallback_response = json.loads(fallback_response_raw)
            # response= fallback_response.get("response", "No reference found")
            # refrence = fallback_response.get("reference", "No reference found")
            # return {
            #     "response": response,
            #     "reference": refrence
            # }

        embeddings = embed_texts(chunks,batch_size=100)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize

        index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
        index.add(embeddings)

        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump((chunks, metadata), f)
            
    if country.lower() not in user_query.lower():
        query_with_context = f"{user_query} in {country}"
    else:
        query_with_context = user_query

    min_similarity = COUNTRY_SIMILARITY_THRESHOLDS.get(country.lower(), 0.62)  # default fallback
    top_chunks = search_faiss_index(query_with_context, index, chunks, metadata, min_similarity=min_similarity)
    # top_chunks = search_faiss_index(query_with_context, index, chunks, metadata)

    # top_chunks = search_faiss_index(user_query, index, chunks, metadata)

    if top_chunks:
        combined = "\n---\n".join([f"From {book}:\n{chunk}" for book, chunk in top_chunks])
        book_names = list(set(book for book, _ in top_chunks))
        
        prompt = (
        f"As a tax assistant for {country}, answer the user's question using the content below. "
        f"Do not mention the source explicitly. Provide a clear and natural answer.\n\n"
        f"If the response is short or simple, answer in a brief paragraph without bullet points or headings.\n"
        f"If the response is long or includes multiple parts or explanations, organize it using bullet points.\n"
        f"Only use headings and subheadings if there are clearly distinct sections or categories in the answer.\n\n"
        f"User Query: {user_query}\n\nRelevant Reference Material:\n{combined}"
    )

        
        # prompt = (
        #         f"As a tax assistant for {country}, answer the user's question using the content below. "
        #         f"Do not mention the source explicitly in your response. Just provide a clear, natural answer:\n\n"
        #         f"User: {user_query}\n\nRelevant Reference Material:\n{combined}"
        #     )

        # prompt = (
        #     f"Use the following content extracted from the book(s) {', '.join(book_names)} "
        #     f"to answer the user query:\n\nUser Query: {user_query}\n\n{combined}"
        # )

        response = query_gpt(prompt, history=history)
        return {
            "response": response,
            "reference": ", ".join(book_names)
        }

    # Fallback if no relevant chunks were found
    print("⚠️ No relevant chunks found. Falling back to general knowledge.")
    fallback_response_raw = get_response_from_general_knowledge(user_query, country, history)
    print("🔥 fallback_response_raw (type):", type(fallback_response_raw))
    print("🔥 fallback_response_raw (value):", fallback_response_raw)
    print("get_response_from_general_knowledge")
    print(f"user query :{user_query}")
    print(f"country:{country}")
    print(f"History:{history}")

    # Safe fallback parsing
    # Determine the type of the fallback
    if isinstance(fallback_response_raw, dict):
        fallback_response = fallback_response_raw
    else:
        try:
            fallback_response = json.loads(fallback_response_raw)
        except (json.JSONDecodeError, TypeError):
            
            return {
                "response": fallback_response_raw,
                "reference": "N/A"
            }

    # Extract response fields from JSON
    response = fallback_response.get("response", "No response found")
    reference = fallback_response.get("reference", "No reference found")

    return {
        "response": response,
        "reference": reference
    }
    # fallback_response = json.loads(fallback_response_raw)
    # response = fallback_response.get("response", "No response found")
    # reference = fallback_response.get("reference", "No reference found")

    # return {
    #     "response": response,
    #     "reference": reference
    # }

    # print(f"DEBUG: fallback_response_raw = {repr(fallback_response_raw)}")
    # print(fallback_response_raw)
    # print(type(fallback_response_raw)) 
    # fallback_response = json.loads(fallback_response_raw)
    # response= fallback_response.get("response", "No reference found")
    # refrence = fallback_response.get("reference", "No reference found")
   
    # return {
    #     fallback_response_raw
    # }


from fastapi import FastAPI, HTTPException
from pathlib import Path


# Specify the directory where .pkl files should be checked and deleted
pkl_directory = Path("./")


@app.post("/delete_pkl_files")
async def delete_pkl_files():
    try:
        # Check if the specified directory exists
        if not pkl_directory.is_dir():
            raise HTTPException(status_code=500, detail="Invalid directory path")

        # Find and delete .pkl files in the directory
        for file_path in pkl_directory.glob("*.pkl"):
            file_path.unlink()

        return {"message": "Successfully deleted .pkl files"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/taxationbot")
async def chat_with_bot(
    user_id: str = Form(...),
    message: str = Form(...),
    country: str = Form(...),
):
    history = get_user_history(user_id)
    history.append({"role": "user", "content": message})

    result = get_response_from_books(message, country, history)
    print("********************************")
    print(f"printing the results:{result}")
    

    history.append({"role": "assistant", "content": result["response"]})
    save_user_history(user_id, history)

    return JSONResponse(content={
        "response": result["response"],
        "reference": result["reference"]
    })


@app.delete("/delete-history/{user_id}")
async def delete_user_history(user_id: str):
    path = os.path.join(SESSIONS_DIR, f"{user_id}.pkl")
    if os.path.exists(path):
        os.remove(path)
        return JSONResponse(
            status_code=200,
            content={"message": f"History for user '{user_id}' has been deleted."}
        )
    else:
        raise HTTPException(status_code=404, detail=f"No history found for user '{user_id}'.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
