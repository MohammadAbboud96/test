from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import streamlit as st
import os

from models.video_summarizer import VideoSummarizer
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# RESTful API Setup
app = FastAPI()
summarizer = VideoSummarizer()

class VideoRequest(BaseModel):
    video_url: str

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/add-video")
def add_video(video: VideoRequest):
    try:
        transcript = summarizer.fetch_transcript(video.video_url)
        summary = summarizer.summarize_transcript(transcript)
        summarizer.store_summary(video.video_url, summary)
        return {"message": "Video summary added successfully", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(query: QueryRequest):
    try:
        response = summarizer.query(query.question, query.chat_history)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit Interface
st.title("Video Summarizer Chatbot")
option = st.selectbox("Choose an action", ["Add Video", "Ask Question"])

if option == "Add Video":
    video_url = st.text_input("Enter YouTube video URL:")
    if st.button("Summarize and Add"):
        try:
            transcript = summarizer.fetch_transcript(video_url)
            summary = summarizer.summarize_transcript(transcript)
            summarizer.store_summary(video_url, summary)
            st.success("Video summary added successfully")
            st.write(summary)
        except Exception as e:
            st.error(str(e))

elif option == "Ask Question":
    question = st.text_input("Enter your question:")
    chat_history = st.session_state.get("chat_history", [])

    if st.button("Ask"):
        try:
            response = summarizer.query(question, chat_history)
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": response})
            st.session_state["chat_history"] = chat_history
            st.write(response)
        except Exception as e:
            st.error(str(e))

# React Frontend (Guidance)
# 1. Create a React app using create-react-app or Vite.
# 2. Add a form for video URL submission and a chat interface.
# 3. Use Axios to interact with the RESTful API.
# Example:
# POST /add-video to upload a video URL.
# POST /ask to ask a question about stored summaries.

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
