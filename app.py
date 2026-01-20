from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-catboot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/test-retrieval", methods=["GET"])
def test_retrieval():
    """Test endpoint to check vector store retrieval"""
    query = request.args.get('query', 'What is diabetes?')
    
    # Test direct retrieval from vector store
    docs = retriever.invoke(query)
    
    result = {
        "query": query,
        "num_docs_retrieved": len(docs),
        "documents": [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get('source', 'Unknown'),
                "full_metadata": doc.metadata
            }
            for doc in docs
        ]
    }
    
    return jsonify(result)


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    
    # Print retrieved documents from vector store
    print("\n" + "="*50)
    print("RETRIEVED DOCUMENTS FROM VECTOR STORE:")
    print("="*50)
    if "context" in response:
        for i, doc in enumerate(response["context"], 1):
            print(f"\n--- Document {i} ---")
            print(f"Content: {doc.page_content[:200]}...")  # Print first 200 chars
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("="*50 + "\n")
    #   end of code for retrieved documents from vs db
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True, use_reloader=False)
