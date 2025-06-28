import os
import uuid
import tempfile
import shutil
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)
CORS(app)

load_dotenv()


UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


active_sessions = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_document(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == 'txt':
        loader = TextLoader(file_path)
    else:
        return None
    
    return loader.load()

def process_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    try:
        collection_name = f"doc-chat-{str(uuid.uuid4())[:8]}"
        persist_directory = os.path.join(UPLOAD_FOLDER, collection_name)
        os.makedirs(persist_directory, exist_ok=True)
        
        embeddings = OpenAIEmbeddings()
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        vector_store.persist()
        
        return vector_store, persist_directory
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def setup_rag_chain(vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_system_prompt = (
        "You are a helpful AI assistant that answers questions about documents. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Be concise but thorough in your response.\n\n"
        "Context: {context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    
    return rag_chain

def cleanup_resources(persist_directory, file_path):
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Cleaned up vector store directory: {persist_directory}")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up uploaded file: {file_path}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'File type not allowed. Use PDF or TXT files.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        
        documents = load_document(file_path)
        if not documents:
            return jsonify({'error': 'Unable to load document'}), 400
        
        chunks = process_document(documents)
        vector_store, persist_directory = create_vector_store(chunks)
        rag_chain = setup_rag_chain(vector_store)
        
        
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            'rag_chain': rag_chain,
            'vector_store': vector_store,
            'persist_directory': persist_directory,
            'file_path': file_path,
            'document_name': filename,
            'chat_history': []
        }
        
        return jsonify({
            'session_id': session_id,
            'document_name': filename,
            'chunk_count': len(chunks),
            'message': f"Successfully uploaded and processed '{filename}'. You can now ask questions about the document."
        })
    
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Failed to process document: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    
    if not data or 'session_id' not in data or 'message' not in data:
        return jsonify({'error': 'Missing required fields: session_id and message'}), 400
    
    session_id = data['session_id']
    user_message = data['message']
    
    if session_id not in active_sessions:
        return jsonify({'error': 'Invalid session ID. Please upload a document first.'}), 404
    
    try:
        session = active_sessions[session_id]
        rag_chain = session['rag_chain']
        vector_store = session['vector_store']
        chat_history = session['chat_history']
        formatted_chat_history = []
        for msg in chat_history:
            formatted_chat_history.append(msg)
        response = rag_chain.invoke({
            "input": user_message,
            "chat_history": formatted_chat_history
        })
        
        answer = response.get("answer", "")
        source_docs = response.get("context", [])
        sources = []
        
        if source_docs:
            for doc in source_docs[:3]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                
                
                if page:
                    source_info = f"{os.path.basename(source)} (Page {page + 1})"
                else:
                    source_info = os.path.basename(source)
                
                if source_info not in sources:
                    sources.append(source_info)
        else:
            docs = vector_store.similarity_search(user_message, k=2)
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                
                if page:
                    source_info = f"{os.path.basename(source)} (Page {page + 1})"
                else:
                    source_info = os.path.basename(source)
                
                if source_info not in sources:
                    sources.append(source_info)
        chat_history.append({"type": "human", "content": user_message})
        chat_history.append({"type": "ai", "content": answer})
        session['chat_history'] = chat_history
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "session_id": session_id
        })
    
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({'error': f'Failed to process message: {str(e)}'}), 500

@app.route('/api/sessions/<session_id>/history', methods=['GET'])
def get_chat_history(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    session = active_sessions[session_id]
    
    return jsonify({
        'session_id': session_id,
        'document_name': session['document_name'],
        'chat_history': session['chat_history']
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def end_session(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Invalid session ID'}), 404

    try:
        session = active_sessions[session_id]
        persist_directory = session['persist_directory']
        file_path = session['file_path']
        
        cleanup_resources(persist_directory, file_path)
        
        del active_sessions[session_id]
        
        return jsonify({'message': 'Session ended successfully'})
    
    except Exception as e:
        print(f"Error during end_session: {str(e)}")
        return jsonify({'error': f'Failed to end session: {str(e)}'}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    sessions = []
    for session_id, session_data in active_sessions.items():
        sessions.append({
            'session_id': session_id,
            'document_name': session_data['document_name'],
            'message_count': len(session_data['chat_history'])
        })
    
    return jsonify({'sessions': sessions})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Ensure you set your openai key email aayushpalai@gmail.com if u need 1")
    app.run(debug=True, host='0.0.0.0', port=5000)