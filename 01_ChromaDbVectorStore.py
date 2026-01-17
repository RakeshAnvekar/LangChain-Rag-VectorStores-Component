from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file




# create langchain documents objects for IPL players

doc1=Document(page_content="Virat Kohli is an Indian cricketer and former captain of the Indian national team.", metadata={"player":"Virat Kohli", "team":"Royal Challengers Bangalore"})
doc2=Document(page_content="Suresh Raina is an Indian cricketer and former captain of the Mumbai Indians team.", metadata={"player":"Suresh Raina", "team":"Mumbai Indians"})
doc3=Document(page_content="MS Dhoni is an Indian cricketer and former captain of the Chennai Super Kings team.", metadata={"player":"MS Dhoni", "team":"Chennai Super Kings"})
doc4=Document(page_content="Rohit Sharma is an Indian cricketer and captain of the Mumbai Indians team.", metadata={"player":"Rohit Sharma", "team":"Mumbai Indians"})
doc5=Document(page_content="Jasprit Bumrah is an Indian cricketer who plays for the Mumbai Indians team.", metadata={"player":"Jasprit Bumrah", "team":"Mumbai Indians"})


docs=[doc1, doc2, doc3, doc4, doc5 ]
vectorstore=Chroma(
embedding_function=OpenAIEmbeddings(),
persist_directory="chroma_db", ## in root foldr we create the folder name with this name to store the vectordb files
collection_name="ipl_players"
)

# add documents to the vectordb
vectorstore.add_documents(docs)# when you add the documets by default id will be generated for each document
vectorstore.persist() # to save the vectordb to the persist_directory
print("Documents added to ChromaDB vectorstore successfully.")

# Now you can query the vectordb to retrieve similar documents
query="Who is the captain of Mumbai Indians?"

result=vectorstore.similarity_search(query, k=1) # k is number of similar documents to retrieve

print("Query:", query)
print("Most similar document:", result[0].page_content)

#similarity score

similarity_scores=vectorstore.similarity_search_with_score(query, k=3)
print("\nTop 3 similar documents with scores:")
for doc, score in similarity_scores:
    print(f"Score: {score:.4f}, Document: {doc.page_content}")

#metadata filtering
filtered_results=vectorstore.similarity_search(query, k=2, filter={"team":"Mumbai Indians"})
print("\nFiltered results (team: Mumbai Indians):")
for doc in filtered_results:
    print(doc.page_content, doc)

#update existing document in vectordb

updated_doc=Document(page_content="Rohit Sharma is an Indian cricketer and the current captain of the Mumbai Indians team.", metadata={"player":"Rohit Sharma", "team":"Mumbai Indians"})
vectorstore.update_documents(updated_docs=updated_doc, ids="4") # assuming id of Rohit Sharma document is "4"
vectorstore.persist()

#delete document from vectordb
vectorstore.delete_documents(ids=["2"]) # assuming id of Suresh Raina document is


