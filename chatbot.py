import mysql.connector
import json
import ollama

from sentence_transformers import SentenceTransformer

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-llm:latest"

llm_agent = ollama.Client(host=OLLAMA_HOST)
embedder = SentenceTransformer('BAAI/bge-m3')

db = mysql.connector.connect(
  host = "gateway01.eu-central-1.prod.aws.tidbcloud.com",
  port = 4000,
  user = "3Kwp8ZSpU7iyhfZ.root",
  password = "6nprUxuPRNEl0OKZ",
  database = "RAG",
  ssl_ca = "/etc/ssl/certs/ca-certificates.crt",
  ssl_verify_cert = True,
  ssl_verify_identity = True
)

def search_document(database, query, k_top=5):
    results = []
    
    query_embedding_list = embedder.encode(query).tolist()
    query_embedding_str =json.dumps(query_embedding_list)
    
    curr = database.cursor()
    
    sql_query = f"""
                    SELECT text, vec_cosine_distance(embedding,%s) AS distance
                    FROM dataset
                    ORDER BY distance ASC
                    LIMIT {k_top}
            """
    
    curr.execute(sql_query, (query_embedding_str,))
    search_results = curr.fetchall()
    database.commit()
    curr.close()
    
    for result in search_results:
        text, distance = result
        results.append({
            'text':text,
            'distance':distance
        })
    
    return results

def response_query(database, query):
    retrieved_doc = search_document(database, query)
    
    context = "\n".join([doc['text'] for doc in retrieved_doc])
    prompt = f"Jawab pertanyaan berikut berdasarkan konteks yang disediakan {context} \n\npertanyaan :{query}"
    response = llm_agent.chat(model=OLLAMA_MODEL, messages=[
        {
            'role': 'user',
            'content':prompt
        }
    ])
    
    return response['message']['content']

if __name__ == "__main__":
    print("Chatbot Dimulai")
    while True:
        query_text = input("Masukkan resep yang ingin anda tahu : ")
        
        if query_text.lower() in ['exit', 'quit', 'q']:
            print("Closing Chatbot....")
            break
        
        response = response_query(database=db, query=query_text)
        print("Resep : ", response)

print("Chatbot Selesai")
