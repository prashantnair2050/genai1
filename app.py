import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Model setup
model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token="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")
pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_length=50, min_length=5, repetition_penalty=1.2)
llm = HuggingFacePipeline(pipeline=pipe)

# Load data
loader = TextLoader("ai_intro.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.split_documents(documents)
# Embeddings with token
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": "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"}
)
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()

# Chain setup
def extract_context(docs):
    return ' '.join(doc.page_content for doc in docs) if docs else 'No context available.'
prompt = PromptTemplate(input_variables=['question', 'chat_history', 'context'], template='Using this: {context}\nBased on our chat: {chat_history}\nAnswer in one sentence: {question}')
chain = {'context': lambda x: extract_context(retriever.invoke(x['question'])), 'question': lambda x: x['question'], 'chat_history': lambda x: x['chat_history']} | prompt | llm
history = InMemoryChatMessageHistory()
conversation = RunnableWithMessageHistory(chain, lambda: history, input_messages_key='question', history_messages_key='chat_history')

# Streamlit interface
st.title("AI Q&A Bot")
question = st.text_input("Ask me anything:")
if question:
    response = conversation.invoke({'question': question}, config={'configurable': {'session_id': '1'}})
    st.write("Bot:", response)
