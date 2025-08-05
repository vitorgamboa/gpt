import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.schema import Document
import os
load_dotenv()

st.set_page_config(page_title="Vamos ver um negócio")
st.title("Vamos ver um negócio")

# Campo para o token de API
api_token = st.text_input("Digite seu token de API", type="password")

if not api_token:
    st.warning("Por favor, insira seu token de API para continuar.")
    st.stop()

id_model = "gemma2-9b-it"
temperature = 0.7
path = "uploads"

# Carregamento da LLM


def load_llm(id_model, temperature, api_token):
    print("Entrou Chatgroq")
    llm = ChatGroq(
        model=id_model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_token  # Passa o token de API aqui
    )
    print("Saiu Chatgroq")
    return llm


llm = load_llm(id_model, temperature, api_token)

# Exibição do resultado


def show_res(res):
    from IPython.display import Markdown, display
    if "</think>" in res:
        res = res.split("</think>")[-1].strip()
    else:
        res = res.strip()  # fallback se não houver tag
    display(Markdown(res))

# Extração do conteúdo


def extract_text_pdf(file_path):
    print("Entrou Extract")
    loader = PyMuPDFLoader(file_path)
    doc = loader.load()
    content = "\n".join([page.page_content for page in doc])
    print("Saiu Extract")
    return content

# Indexação e recuperação


def config_retriever(folder_path="uploads"):
    print("Entrou retriever")

    # Caminho para o índice FAISS
    index_path = 'index_faiss'

    # Embeddings
    embedding_model = "BAAI/bge-m3"  # sentence-transformers/all-mpnet-base-v2

    print("aqui 1")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Verifica se o índice já existe
    if os.path.exists(index_path):
        # Carregar o índice FAISS existente em vez de reprocessar
        vectorstore = FAISS.load_local(
            index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        st.success("Índice FAISS carregado com sucesso!")
    else:
        # Carregar documentos
        docs_path = Path("uploads")
        pdf_files = [f for f in docs_path.glob("*.pdf")]

        if len(pdf_files) < 1:
            st.error("Nenhum arquivo PDF carregado")
            st.stop()

        loaded_documents = [extract_text_pdf(pdf) for pdf in pdf_files]

        # Divisão em pedaços de texto / Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = []
        for doc in loaded_documents:
            print("Entrou for loaded")
            chunks.extend(text_splitter.split_text(doc))
            print("Saiu for loaded")

        print("aqui 2")
        # Criação de documentos a partir de chunks
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Carregar e processar os documentos aqui
        vectorstore = FAISS.from_documents(
            documents=documents, embedding=embeddings)
        # vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        print("aqui 3")
        # Armazenamento
        vectorstore.save_local(index_path)
        st.success("Índice FAISS criado e salvo com sucesso!")
        print("aqui 4")

    # Configurando o recuperador de texto / Retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 4}
    )

    print("Saiu retriever")
    return retriever


# Chain da RAG


def config_rag_chain(llm, retriever):
    print("Entrou rag chain")
    # Prompt de contextualização
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

    context_q_system_prompt = context_q_system_prompt
    context_q_user_prompt = "Question: {input}"
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )

    # Chain para contextualização
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=context_q_prompt
    )

    # Prompt para perguntas e respostas (Q&A)
    system_prompt = """Você é um assistente virtual prestativo e está respondendo perguntas gerais sobre os serviços de uma empresa.
  Use os seguintes pedaços de contexto recuperado para responder à pergunta.
  Se você não sabe a resposta, apenas comente que não sabe dizer com certeza.
  Mas caso seja uma dúvida muito comum, pode sugerir como alternativa uma solução possível.
  Mantenha a resposta concisa.
  Responda em português. \n\n"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\n Contexto: {context}"),
    ])

    # Configurar LLM e Chain para perguntas e respostas (Q&A)

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )
    print("Saiu rag chain")
    return rag_chain

# Interação com chat


def chat_llm(rag_chain, input):

    st.session_state.chat_history.append(HumanMessage(content=input))

    response = rag_chain.invoke({
        "input": input,
        "chat_history": st.session_state.chat_history
    })

    res = response["answer"]
    res = res.split(
        "</think>")[-1].strip() if "</think>" in res else res.strip()

    st.session_state.chat_history.append(AIMessage(content=res))

    return res


input = st.chat_input("Digite sua mensagem aqui...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Olá, sou um assistente virtual! Como posso te ajudar?"),
    ]

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

if input is not None:
    with st.chat_message("Human"):
        st.markdown(input)

    with st.chat_message("AI", avatar="🤖"):
        if st.session_state.retriever is None:
            st.session_state.retriever = config_retriever(path)
        rag_chain = config_rag_chain(llm, st.session_state.retriever)
        res = chat_llm(rag_chain, input)
        st.write(res)
