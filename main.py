import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client with API key from environment variable
client = Groq(api_key=GROQ_API_KEY)

# Define the ChatBuffer class with a rolling buffer
class ChatBuffer:
    def __init__(self, file_path, log_file_path, max_size=5):
        self.buffer = {}  # Dictionary to store session histories
        self.file_path = file_path
        self.log_file_path = log_file_path
        self.max_size = max_size  # Maximum number of conversation pairs to store
        self.load_from_file()

    def add_message_pair(self, session_id, question, answer):
        if session_id not in self.buffer:
            self.buffer[session_id] = deque(maxlen=self.max_size)
        # Add new question-answer pair to the buffer (deque)
        self.buffer[session_id].append({"question": question, "answer": answer})
        self.save_to_file()
        self.log_interaction(session_id, question, answer)

    def get_conversation(self, session_id):
        return list(self.buffer.get(session_id, []))

    def save_to_file(self):
        with open(self.file_path, 'w') as file:
            json.dump({k: list(v) for k, v in self.buffer.items()}, file)

    def load_from_file(self):
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                # Convert lists back to deques
                self.buffer = {k: deque(v, maxlen=self.max_size) for k, v in data.items()}
        except FileNotFoundError:
            self.buffer = {}

    def log_interaction(self, session_id, question, answer):
        with open(self.log_file_path, 'a') as log_file:
            timestamp = datetime.now().isoformat()
            log_entry = {"timestamp": timestamp, "session_id": session_id, "question": question, "answer": answer}
            log_file.write(json.dumps(log_entry) + '\n')

# Initialize the buffer with a maximum size of 5
buffer = ChatBuffer(file_path='chat_buffer.json', log_file_path='chat_log.json', max_size=5)

# Define the FastAPI app
app = FastAPI()

# Define the request model
class ChatRequest(BaseModel):
    message: str
    session_id: str

# Define the response model
class ChatResponse(BaseModel):
    response: str

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat_with_groq(request: ChatRequest):
    try:
        session_id = request.session_id

        # Retrieve the conversation history
        conversation_history = buffer.get_conversation(session_id)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add the last 5 conversations to the message history
        for pair in conversation_history:
            messages.append({"role": "user", "content": pair["question"]})
            messages.append({"role": "assistant", "content": pair["answer"]})

        # Add the new message from the user
        messages.append({"role": "user", "content": request.message})

        # Perform the chat completion
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
        )
        response_message = chat_completion.choices[0].message.content

        # Add the new question-answer pair to the buffer
        buffer.add_message_pair(session_id, request.message, response_message)

        return ChatResponse(response=response_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Groq Chat API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
