## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The objective is to create a chatbot that can intelligently respond to queries based on information extracted from a PDF document. By using LangChain, the chatbot will be able to process the content of the PDF and use a language model to provide relevant answers to user queries. The effectiveness of the chatbot will be evaluated by testing it with various questions related to the document.

### DESIGN STEPS:
1)Initialization:
Input: PDF document path.

Output: Document loader, embeddings, vector database, prompt, and chain.

2)Load PDF Content:

Import and initialize the PyPDFLoader with the provided PDF file path.
Extract the content of the PDF into pages.

3)Embed the Document Chunks:

Import and initialize OpenAIEmbeddings to generate embeddings.

Initialize the Chroma vector database with:
Persistent storage directory (persist_directory).
The embedding function.

4)Define the Language Model:
Import and initialize ChatOpenAI with:

Model name (gpt-4).
Temperature (0) for deterministic responses.
5)Create a Retrieval Prompt:
Define a concise, user-friendly prompt template to:

Use context from the document.

Limit answers to three sentences.

Encourage polite responses with "Thanks for asking!" at the end.

6) Build the Retrieval Chain:
Initialize the RetrievalQA chain by:

Specifying the language model (llm).
Linking the retriever (vectordb.as_retriever()).
Applying the prompt template.
Enabling source document return for transparency.
7)Run the Query:
Take a query (question) as input.

Pass the query to the qa_chain for processing.

Retrieve the result and its associated source documents.

8) Output the Result:
Print the query (question).

Print the chatbot’s answer (result["result"]).

### PROGRAM:
```
# Step 1: Install necessary packages (run in Jupyter or terminal)
# !pip install langchain openai pypdf chromadb tiktoken python-dotenv

# Step 2: Import libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

# Step 3: Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 4: Load and split the PDF
loader = PyPDFLoader("ml_lecture.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

# Step 5: Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents, embeddings)

# Step 6: Create QA chain with retriever
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# Step 7: Query the chatbot
query = "What is the main topic discussed in the document?"
response = qa_chain.run(query)
print("Answer:", response)
```

### OUTPUT:

### RESULT:
