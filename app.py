import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.prompt import system_prompt


app = Flask(__name__, template_folder="template")

load_dotenv(Path(__file__).resolve().with_name(".env"))

required_keys = ("PINECONE_API_KEY", "OPENAI_API_KEY")
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(missing_keys)}"
    )

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_name = "medi-assist"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_completion_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_retrieval = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_retrieval)


@app.get("/")
def home():
    return render_template("chat.html")


@app.post("/get")
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify(error="Please enter a message."), 400

    response = rag_chain.invoke({"input": msg})
    return jsonify(answer=str(response["answer"]))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)



