# Databricks notebook source
# MAGIC %md
# MAGIC Run `config` and create helper functions

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-vectorsearch databricks-genai-inference llama-index llama-index-readers-web
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config $reset_all_data=false

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
from typing import Iterator
from mlflow import MlflowClient
import time

# COMMAND ----------

#Temp workaround to test if a table exists in shared cluster mode in DBR 14.2 (see SASP-2467)
def table_exists(table_name):
    try:
        spark.table(table_name).isEmpty()
    except:
        return False
    return True

# COMMAND ----------


def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" not in str(e):
            print(
                f"Unexpected error describing the index. This could be a permission issue."
            )
            raise e
    return False

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get("status", idx.get("index_status", {}))
        status = index_status.get(
            "detailed_state", index_status.get("status", "UNKNOWN")
        ).upper()
        url = index_status.get("index_url", index_status.get("url", "UNKNOWN"))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            print(
                f"Can't get the status - will assume index is ready {idx} - url: {url}"
            )
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                print(
                    f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}"
                )
            time.sleep(10)
        else:
            raise Exception(
                f"""Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}"""
            )
    raise Exception(
        f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}"
    )

# COMMAND ----------

# from llama_index.readers.web import SimpleWebPageReader
# from llama_index.core.node_parser import SentenceSplitter
# from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType 

# # llamaindex tools for getting and splitting text
# reader = SimpleWebPageReader(html_to_text=True)
# parser = SentenceSplitter.from_defaults() 

# # Text chunks for splitting 
# chunks = parser.get_nodes_from_documents(reader.load_data(urls))

# # The schema of our source delta table 
# schema = StructType(
#     [
#         StructField("id", StringType(), True),
#         StructField("url", StringType(), True),
#         StructField("text", StringType(), True),
#     ]
# )

# # Initialize an empty DataFrame with the defined schema 
# df = spark.createDataFrame([], schema)

# # Iterate through chunks 
# for chunk in chunks: 
#     chunk = chunk.dict()
#     chunk_id = chunk["id_"]
#     chunk_url = chunk["url"]
#     chunk_text = chunk["text"]

#     # new_row = spark.createDataFrame([(chunk_id, chunk_text)], schema)
#     new_row = spark.createDataFrame([(chunk_id, chunk_url, chunk_text)], schema)
#     df = df.union(new_row)

# df.write.format("delta").mode("append").saveAsTable("databricks_generative_ai_world_cup.netflix.source_table")
# display(spark.table("databricks_generative_ai_world_cup.netflix.source_table"))

# COMMAND ----------

# import requests
# from bs4 import BeautifulSoup
# import xml.etree.ElementTree as ET
# from concurrent.futures import ThreadPoolExecutor
# from pyspark.sql.types import StringType
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# from llama_index.readers.web import SimpleWebPageReader
# from llama_index.core.node_parser import SentenceSplitter
# from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType 

# # llamaindex tools for getting and splitting text
# reader = SimpleWebPageReader(html_to_text=True)
# parser = SentenceSplitter.from_defaults() 

# # Add retries with backoff to avoid 429 while fetching the doc
# retries = Retry(
#     total=100,
#     backoff_factor=0.1, 
#     status_forcelist=[429],
# )

# def download_netflix_documentation_articles(max_documents=None):
#     # Fetch the XML content from sitemap
#     response = requests.get(NETFLIX_SITEMAP_URL)
#     root = ET.fromstring(response.content)

#     # Find all 'loc' elements (URLs) in the XML
#     urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
#     if max_documents:
#         urls = urls[:max_documents]

#     # Create DataFrame from URLs
#     df_urls = spark.createDataFrame(urls, StringType()).toDF("url").repartition(10)
#     print("df_urls", df_urls.count(), df_urls.head(1))

#     # Pandas UDF to fetch HTML content for a batch of URLs
#     @pandas_udf("string")
#     def fetch_html_udf(urls: pd.Series) -> pd.Series:
#         adapter = HTTPAdapter(max_retries=retries)
#         http = requests.Session()
#         http.mount("http://", adapter)
#         http.mount("https://", adapter)
#         def fetch_html(url):
#             # return reader.load_data(url)
#             try:
#                 response = http.get(url)
#                 if response.status_code == 200:
#                     return response.content
#             except requests.RequestException:
#                 return None
#             return None

#         with ThreadPoolExecutor(max_workers=200) as executor:
#             results = list(executor.map(fetch_html, urls))
#         return pd.Series(results)

#     # Pandas UDF to process HTML content and extract text
#     @pandas_udf("string")
#     def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
#         def extract_text(html_content):
#             # return parser.get_nodes_from_documents(html_content)
#             if html_content:
#                 soup = BeautifulSoup(html_content, "html.parser")
#                 article_div = soup.find("div", itemprop="articleBody")
#                 if article_div:
#                     return str(article_div).strip()
#             return None

#         return html_contents.apply(extract_text)

#     # Apply UDFs to DataFrame
#     df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
#     print("df_with_html", df_with_html.count(), df_with_html.head(1))
#     final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))
#     print("final_df 1", final_df.count(), final_df[0].url, final_df[0].text[:100])
#     # print("final_df 1", final_df.count(), final_df.head(1))

#     # Select and filter non-null results
#     final_df = final_df.select("url", "text").filter("text IS NOT NULL").cache()
#     print("final_df 2", final_df.count(), final_df.head(1))

#     if final_df.isEmpty():
#       raise Exception("Dataframe is empty, couldn't download Netflix documentation, please check sitemap status.")

#     return final_df
# df = download_netflix_documentation_articles(10)
# df.head()

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.types import StringType
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter

# llamaindex tools for getting and splitting text
reader = SimpleWebPageReader(html_to_text=True)
parser = SentenceSplitter.from_defaults() 

# DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"

# Add retries with backoff to avoid 429 while fetching the doc
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429],
)

def download_netflix_documentation_articles(max_documents=None):
    # Fetch the XML content from sitemap
    response = requests.get(NETFLIX_SITEMAP_URL)
    root = ET.fromstring(response.content)

    # Find all 'loc' elements (URLs) in the XML
    urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if max_documents:
        urls = urls[:max_documents]

    print(urls)

    # Create DataFrame from URLs
    df_urls = spark.createDataFrame(urls, StringType()).toDF("url").repartition(10)

    # Pandas UDF to fetch HTML content for a batch of URLs
    @pandas_udf("string")
    def fetch_html_udf(urls: pd.Series) -> pd.Series:
        adapter = HTTPAdapter(max_retries=retries)
        http = requests.Session()
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        def fetch_html(url):
            try:
                response = http.get(url)
                if response.status_code == 200:
                    return response.content
            except requests.RequestException:
                return None
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = list(executor.map(fetch_html, urls))
        return pd.Series(results)

    # Pandas UDF to process HTML content and extract text
    @pandas_udf("string")
    def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
        def extract_text(html_content):
            return str(parser.get_nodes_from_documents(html_content)).strip()
            # print('start of the function')
            # if html_content:
            #     # return reader.load_data(urls)
            #     print("there's html content", html_content[:10])
            #     soup = BeautifulSoup(html_content, "html.parser")
            #     article_div = soup.find("div", itemprop="articleBody")
            #     if article_div:
            #         return str(article_div).strip()
            # print("returned none")
            # return None
        
        return html_contents.apply(extract_text)

    # Apply UDFs to DataFrame
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
    print("df_with_html", df_with_html.count(), df_with_html.head())
    final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))
    print("final_df 1", final_df.count(), final_df.head(1))

    # Select and filter non-null results
    final_df = final_df.select("url", "text").filter("text IS NOT NULL").cache()
    print("final_df 2", final_df.count(), final_df.head(1))
    if final_df.isEmpty():
      raise Exception("Dataframe is empty, couldn't download Netflix documentation, please check sitemap status.")

    return final_df

df = download_netflix_documentation_articles(10)
df.head()

# COMMAND ----------


