import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
import os
import re

# Import HTML templates
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

# Set the page configuration
st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
    return text

def extract_code_snippets(text):
    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    return code_pattern.findall(text)

def text_splitter(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=600,
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    return text_splitter.split_text(text=raw_text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, retriever):
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        st.error("COHERE_API_KEY not found. Please set the environment variable.")
        return "Error: API key not found."

    cohere_llm = Cohere(
        model="command",
        temperature=0.1,
        cohere_api_key=cohere_api_key,
        max_tokens=2000
    )

    prompt_template = """Answer the question as precisely and thoroughly as possible using the provided context. If the answer is
                        not contained in the context, say "answer not available in context". \n\n
                        Context: \n {context} \n\n
                        Question: \n {question} \n
                        Answer:"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    try:
        return rag_chain.invoke(question)
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Error generating answer."

def main():
    st.title("Chat with Multiple PDFs 📄")

    st.sidebar.header("Upload Your PDFs")
    pdf_files = st.sidebar.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')

    if st.sidebar.button("Process"):
        if pdf_files:
            with st.spinner("Processing PDFs..."):
                raw_text = extract_text_from_pdfs(pdf_files)
                text_chunks = text_splitter(raw_text)
                vectorstore = get_vector_store(text_chunks)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                code_snippets = extract_code_snippets(raw_text)
                st.session_state['code_snippets'] = code_snippets
                st.session_state['retriever'] = retriever
                st.success("PDFs processed successfully. You can now ask questions.")
                if code_snippets:
                    st.write("Extracted Code Snippets:")
                    for snippet in code_snippets:
                        st.code(snippet)
        else:
            st.warning("Please upload at least one PDF file.")

    if 'retriever' in st.session_state:
        st.markdown(css, unsafe_allow_html=True)

        chat_html = "<div id='chat-box'>"
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        user_message = st.text_input("You:", key="user_input")
        if st.button("Send"):
            if user_message:
                st.session_state.messages.append({"role": "user", "content": user_message})
                with st.spinner("Generating answer..."):
                    answer = generate_answer(user_message, st.session_state['retriever'])
                    st.session_state.messages.append({"role": "bot", "content": answer})
            else:
                st.warning("Please enter a message.")

        for message in st.session_state.messages:
            if message['role'] == 'user':
                chat_html += user_template.replace('{{MSG}}', message['content'])
            elif message['role'] == 'bot':
                chat_html += bot_template.replace('{{MSG}}', message['content'])

        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    else:
        st.info("Please process the PDFs first by uploading them and clicking 'Process'.")

if __name__ == "__main__":
    main()
