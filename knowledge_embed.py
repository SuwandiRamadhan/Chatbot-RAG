import json
import pandas as pd
import mysql.connector

from sentence_transformers import SentenceTransformer

# Buat instance untuk embedder
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

curr = db.cursor()

# baca dataset_baru
df = pd.read_csv("dataset_baru.csv")

for index, row in df.iterrows():
    text = str(row['Title']) + " " + str(row['Ingredients'])
    
    try:
        embedding_list = embedder.encode(text).tolist()
        embedding_str =json.dumps(embedding_list)
        
        sql_query = """
                        INSERT INTO dataset (text, embedding) VALUES (%s, %s)
                """
        curr.execute(sql_query, (text, embedding_str))
        print(f"Data berhasil index ke{index} berhasil ditambahkan")
    except Exception as e:
        print(f"Data index ke{index} error: {e}")
        print(f"Data index ke{index} gagal ditambahkan: {e}")
        
db.commit()
curr.close()
print("Semua data berhasil ditambahkan")