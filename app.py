# app.py

import streamlit as st
import mysql.connector
import json
import ollama
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_models_and_db():
    print("Memuat model dan koneksi database...")
    
    # Inisialisasi LLM Agent (Ollama)
    llm_agent = ollama.Client(host="http://localhost:11434")
    
    # Inisialisasi Embedder
    embedder = SentenceTransformer('BAAI/bge-m3')
    
    # Koneksi ke Database TiDB Cloud
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
    print("Model dan database berhasil dimuat.")
    return llm_agent, embedder, db

# Memuat semuanya sekali saat aplikasi pertama kali dijalankan
llm_agent, embedder, db = load_models_and_db()

def search_document(database, query, k_top=5):
    """Fungsi untuk mencari dokumen relevan di database."""
    results = []
    
    query_embedding_list = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding_list)
    
    # Menggunakan cursor baru untuk setiap query agar thread-safe
    curr = database.cursor()
    
    sql_query = f"""
                    SELECT text, vec_cosine_distance(embedding,%s) AS distance
                    FROM dataset
                    ORDER BY distance ASC
                    LIMIT {k_top}
            """
    
    curr.execute(sql_query, (query_embedding_str,))
    search_results = curr.fetchall()
    
    database.ping(reconnect=True) 
    curr.close()
    
    for result in search_results:
        text, distance = result
        results.append({
            'text': text,
            'distance': distance
        })
    
    return results

def response_query(database, query):
    # Fungsi untuk mendapatkan jawaban dari LLM berdasarkan dokumen yang ditemukan
    retrieved_doc = search_document(database, query)
    
    if not retrieved_doc:
        return "Maaf, saya tidak dapat menemukan resep yang sesuai dengan permintaan Anda di dalam database saya."
        
    context = "\n".join([doc['text'] for doc in retrieved_doc])
    prompt = f"Anda adalah seorang chef ahli masakan Indonesia. Jawab pertanyaan berikut HANYA berdasarkan konteks resep yang disediakan. Jangan menambahkan informasi dari luar konteks. \n\nKonteks Resep:\n{context} \n\nPertanyaan: {query}\n\nJawaban:"
    
    response = llm_agent.chat(model="deepseek-llm:latest", messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])
    
    return response['message']['content']


# UI Streamlit

st.title("🍳 Resep Masakan Indonesia")
st.caption("Dibangun dengan Arsitektur RAG, Ollama (Deepseek)")

# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan pesan-pesan dari riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Menerima input dari pengguna
if prompt := st.chat_input("Mau masak apa hari ini?"):
    # Tambahkan pesan pengguna ke riwayat chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Tampilkan pesan pengguna di container chat
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Tampilkan pesan dari asisten (bot)
    with st.chat_message("assistant"):
        # Tampilkan status "sedang berpikir..."
        with st.spinner("Mencari resep, silahkan tunggu ..."):
            response = response_query(database=db, query=prompt)
            st.markdown(response)
            
    # Tambahkan respon dari asisten ke riwayat chat
    st.session_state.messages.append({"role": "assistant", "content": response})