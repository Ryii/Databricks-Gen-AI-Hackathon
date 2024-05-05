# Databricks notebook source
# MAGIC %md
# MAGIC # Netflix Chatbot

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-vectorsearch databricks-genai-inference llama-index llama-index-readers-web
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index

# COMMAND ----------

if not table_exists("raw_documentation") or spark.table("raw_documentation").isEmpty():
    # Download Netflix documentation to a DataFrame 
    doc_articles = download_netflix_documentation_articles()
    doc_articles.write.mode('overwrite').saveAsTable("raw_documentation")
display(spark.table("raw_documentation").limit(5))

# COMMAND ----------

# %sql
# -- DROP TABLE IF EXISTS databricks_generative_ai_world_cup.netflix.source_table;
# -- DROP TABLE IF EXISTS databricks_generative_ai_world_cup.netflix.vs_index;

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2, but merge small h2 chunks together to avoid too small. 
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=500):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # Merge chunks together to add text before h2 and avoid too small docs.
  for c in h2_chunks:
    # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # Discard too small chunks
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]
 
# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType 

# Create schema and table 
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {CATALOG}.{DB}')
spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{DB}.netflix_documentation (
        id STRING,
        url STRING,
        text STRING
    )
    USING delta 
    TBLPROPERTIES ("delta.enableChangeDataFeed" = "true")
    """
)

# COMMAND ----------

# Chunk all documents with spark
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    
(spark.table("raw_documentation")
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable("netflix_documentation"))

display(spark.table("netflix_documentation"))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient 
vsc = VectorSearchClient()

# COMMAND ----------

# Create endpoint 
if not endpoint_exists(vsc, VS_ENDPOINT):
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
wait_for_vs_endpoint_to_be_ready(vsc, VS_ENDPOINT)
print(f"Endpoint {VS_ENDPOINT} is ready.")

# COMMAND ----------

# Create index 
if not index_exists(vsc, VS_ENDPOINT, VS_INDEX):
  print(f"Creating index {VS_INDEX} on endpoint {VS_ENDPOINT}...")
  vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=VS_INDEX,
    source_table_name=VS_SOURCE_TABLE,
    pipeline_type="CONTINUOUS",
    primary_key="id",
    embedding_source_column="text",
    embedding_model_endpoint_name="databricks-bge-large-en"
  )
wait_for_index_to_be_ready(vsc, VS_ENDPOINT, VS_INDEX)
print(f"Index {VS_INDEX} is ready.")

# COMMAND ----------

# # Remove data from source table 
# spark.sql(
#     "TRUNCATE TABLE databricks_generative_ai_world_cup.netflix.source_table"
# )

# COMMAND ----------

urls=[
  "https://help.netflix.com/en/node/412",
  "https://help.netflix.com/en/node/22",
  "https://help.netflix.com/en/node/407",
  "https://help.netflix.com/en/node/116380",
  "https://help.netflix.com/en/node/244",
  "https://help.netflix.com/en/node/123277",
  "https://help.netflix.com/en/node/13243",
  "https://help.netflix.com/en/node/41049",
  "https://help.netflix.com/en/node/32950",
  "https://help.netflix.com/en/node/113408",
  "https://help.netflix.com/en/node/54896",
  "https://help.netflix.com/en/node/59095",
  "https://help.netflix.com/en/node/29",
#   "https://help.netflix.com/en/node/34",
#   "https://help.netflix.com/en/node/2065",
#   "https://help.netflix.com/en/node/1019",
#   "https://help.netflix.com/en/node/124410",
#   "https://help.netflix.com/en/node/14424",
#   "https://help.netflix.com/en/node/12232",
#   "https://help.netflix.com/en/node/59985",
#   "https://help.netflix.com/en/node/45117",
#   "https://help.netflix.com/en/node/95",
#   "https://help.netflix.com/en/node/11634",
#   "https://help.netflix.com/en/node/100242",
#   "https://help.netflix.com/en/node/100400",
#   "https://help.netflix.com/en/node/470",
#   "https://help.netflix.com/en/node/14384",
#   "https://help.netflix.com/en/node/127429",
#   "https://help.netflix.com/en/node/57688",
#   "https://help.netflix.com/en/node/2090",
#   "https://help.netflix.com/en/node/11634",
#   "https://help.netflix.com/en/node/10421",
#   "https://help.netflix.com/en/node/13245",
#   "https://help.netflix.com/en/node/122698",
#   "https://help.netflix.com/en/node/24853",
#   "https://help.netflix.com/en/node/13444",
#   "https://help.netflix.com/en/node/113300",
#   "https://help.netflix.com/en/node/10523",
#   "https://help.netflix.com/en/node/116022",
#   "https://help.netflix.com/en/contactus",
#   "https://help.netflix.com/en/node/54816",
#   "https://help.netflix.com/en/node/47765",
#   "https://help.netflix.com/en/node/22205",
#   "https://help.netflix.com/en/node/115312",
#   "https://help.netflix.com/en/node/264",
#   "https://help.netflix.com/en/node/114277",
#   "https://help.netflix.com/en/node/114276",
#   "https://help.netflix.com/en/node/114275",
#   "https://help.netflix.com/en/node/121442",
#   "https://help.netflix.com/en/node/121924",
#   "https://help.netflix.com/en/node/412",
#   "https://help.netflix.com/en/node/102377",
#   "https://help.netflix.com/en/node/24926",
#   "https://help.netflix.com/en/node/123279",
#   "https://help.netflix.com/en/node/30081",
#   "https://help.netflix.com/en/node/101653",
#   "https://help.netflix.com/en/node/23939",
#   "https://help.netflix.com/en/node/33222",
#   "https://help.netflix.com/en/node/113160",
#   "https://help.netflix.com/en/titlerequest?ui_action=title-suggestion-quicklinks",
#   "https://help.netflix.com/en/node/14422/CA",
#   "https://help.netflix.com/en/node/2064",
#   "https://help.netflix.com/en/node/60541",
#   "https://help.netflix.com/en/node/47765",
#   "https://help.netflix.com/en/node/115312",
#   "https://help.netflix.com/en/node/24853",
#   "https://help.netflix.com/en/node/129273",
#   "https://help.netflix.com/en/node/128339",
#   "https://help.netflix.com/en/node/54816",
#   "https://help.netflix.com/en/node/124418",
#   "https://help.netflix.com/en/node/62990",
#   "https://help.netflix.com/en/node/68489",
#   "https://help.netflix.com/en/node/68771",
#   "https://help.netflix.com/en/node/110084",
#   "https://help.netflix.com/en/node/365",
#   "https://help.netflix.com/en/node/25970",
#   "https://www.netflix.com/ca/",
#   "https://help.netflix.com/en/node/14361",
#   "https://help.netflix.com/en/node/23934",
#   "https://help.netflix.com/en/node/23887",
#   "https://help.netflix.com/en/node/23924",
#   "https://help.netflix.com/en/node/23904",
#   "https://help.netflix.com/en/node/23876",
#   "https://help.netflix.com/en/node/23878",
#   "https://help.netflix.com/en/node/23932",
#   "https://help.netflix.com/en/node/23931",
#   "https://help.netflix.com/en/node/30081",
#   "https://help.netflix.com/en/node/23931",
#   "https://help.netflix.com/en/node/23888",
#   "https://help.netflix.com/en/node/23888",
]

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2, but merge small h2 chunks together to avoid too small. 
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=500):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # Merge chunks together to add text before h2 and avoid too small docs.
  for c in h2_chunks:
    # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # Discard too small chunks
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]
 
# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter

# llamaindex tools for getting and splitting text
reader = SimpleWebPageReader(html_to_text=True)
parser = SentenceSplitter.from_defaults() 

# Text chunks for splitting 
chunks = parser.get_nodes_from_documents(reader.load_data(urls))

# The schema of our source delta table 
schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("url", StringType(), True),
        StructField("text", StringType(), True),
    ]
)

# Initialize an empty DataFrame with the defined schema 
df = spark.createDataFrame([], schema)

# Iterate through chunks 
for chunk in chunks: 
    chunk = chunk.dict()
    chunk_id = chunk["id_"]
    chunk_url = chunk["url"]
    chunk_text = chunk["text"]

    # new_row = spark.createDataFrame([(chunk_id, chunk_text)], schema)
    new_row = spark.createDataFrame([(chunk_id, chunk_url, chunk_text)], schema)
    df = df.union(new_row)

df.write.format("delta").mode("append").saveAsTable("databricks_generative_ai_world_cup.netflix.source_table")
display(spark.table("databricks_generative_ai_world_cup.netflix.source_table"))

# COMMAND ----------

reader.load_data(urls)

# COMMAND ----------

# MAGIC %md
# MAGIC # Vector Search and Foundation Model 
# MAGIC

# COMMAND ----------

from databricks_genai_inference import ChatSession, Embedding
from databricks.vector_search.client import VectorSearchClient 

class RAG:
  def __init__(
    self, 
    model="databricks-dbrx-instruct",
    system_message="You are a helpful assistant. Answer the user's question. If context is provided, you must answer based on this context.",
    max_tokens=2048,
    index_name=VS_INDEX,
    endpoint=VS_ENDPOINT,
  ):
    self.chat_session = ChatSession(
      model=model,
      system_message=system_message,
      max_tokens=max_tokens,
    )
    self.vsc = VectorSearchClient(disable_notice=True)
    self.endpoint = endpoint
    self.index_name = index_name 
  
  def query_index(self, query, num_results=3):
    """
    Queries the vector search index to retrieve relevant passages based on the given query.
    Returns the concatenated text of the retrieved passages.
    """
    index = self.vsc.get_index(
      endpoint_name=self.endpoint,
      index_name=self.index_name,
    )

    query_response = index.similarity_search(
      query_text="Represent this sentence for searching relevant passages: " + query,
      columns=["text"],
      num_results=num_results,
    )

    if "data_array" not in query_response["result"]:
      return "Sorry, I can't help you with that. Can you try rephrasing the question?"
    
    results = query_response["result"]["data_array"]
    concatenated_text = ""

    for i, result in enumerate(results):
      concatenated_text += result[0]
      if i < len(results) - 1: 
        concatenated_text += "\n---\n"
    
    return concatenated_text

  def query_rag(self, user_input):
    """
    Performs Retrieval-Augmented Generation (RAG) using the provided user input 
    Retrieves relevant context from the index and generates a response using the chat model.
    """
    ctx_chunks = self.query_index(user_input)
    ctx = ("Answer the question baed on the provided context. Context: \n\n" + ctx_chunks + "\n\n")
    self.chat_session.reply(ctx + "\n\n" + user_input)
    bot_response = self.chat_session.last.strip() 
    return bot_response

  def clear_chat(self): 
    """
    Clears the chat session history 
    """
    self.chat_session.history.clear() 

# COMMAND ----------

rag = RAG() 

# COMMAND ----------

print(rag.query_rag("How can I add a method of payment?"))

# COMMAND ----------

print(rag.query_rag("How can I check when my payment is due?"))

# COMMAND ----------

rag.clear_chat() 

# COMMAND ----------

print(rag.query_rag("How do I change my plan?"))

# COMMAND ----------


