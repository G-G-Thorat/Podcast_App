import streamlit as st
import pandas as pd
import os
import re
import openai
from gtts import gTTS
import faiss
import numpy as np
from pydub import AudioSegment
import pandas as pd
import os
from dotenv import load_dotenv
 
openai.api_key = os.environ.get("OPENAI_API_KEY")

def split_audio_by_size(input_file, chunk_size_mb=20):
    """
    Splits an audio file into multiple parts of approximately `chunk_size_mb` MB each.
    Returns a list of chunk file paths.
    """
    audio = AudioSegment.from_file(input_file)
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)

    estimated_duration = len(audio) * (chunk_size_mb / file_size_mb)

    chunks = []
    for i, start_time in enumerate(range(0, len(audio), int(estimated_duration))):
        chunk = audio[start_time:start_time + int(estimated_duration)]
        chunk_filename = f"chunk_{i+1}.mp3"
        chunk.export(chunk_filename, format="mp3")
        chunks.append(chunk_filename)

    return chunks

#input_file = "podcast.mp3"  # Your large podcast file
#chunk_files = split_audio_by_size(input_file)

def transcribe_audio_chunks(chunk_files):
    """
    Transcribes each chunk separately and merges results.
    """
    full_transcript = ""

    for chunk in chunk_files:
        try:
            with open(chunk, "rb") as audio_file:
                transcription = openai.Audio.transcribe("whisper-1", audio_file)
                full_transcript += transcription["text"] + " "  # Append transcript
                #print(f"✅ Transcribed: {chunk}")
        except Exception as e:
            print(f"❌ Error processing {chunk}: {e}")

    return full_transcript.strip()
 
#final_transcript = transcribe_audio_chunks(chunk_files)
#with open("final_transcript.txt", "w", encoding="utf-8") as f:
#        f.write(final_transcript)

#print("\n✅ Full podcast transcription saved as 'final_transcript.txt'")

def clean_transcript(transcript):
    """
    Removes filler words, unnecessary speaker tags, and other noise.
    """
    # Remove filler words (extend list if needed)
    filler_words = ["um", "uh", "you know", "like"]
    for filler in filler_words:
        transcript = re.sub(r"\b" + filler + r"\b", "", transcript)

    # Remove extra spaces and line breaks
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    
    return transcript

#with open("final_transcript.txt", "r", encoding="utf-8") as f:
#    raw_transcript = f.read()

#transcript = clean_transcript(raw_transcript)

def split_transcript(transcript, max_length=500):
    """
    Splits the full transcript into smaller segments for better processing.
    Each segment is around max_length characters.
    """
    # Break the transcript into sentences using punctuation as a delimiter
    sentences = re.split(r'(?<=[.!?])\s+', transcript)

    segments = []
    current_segment = ""

    for sentence in sentences:
        if len(current_segment) + len(sentence) < max_length:
            current_segment += " " + sentence
        else:
            segments.append(current_segment.strip())
            current_segment = sentence  # Start new segment

    if current_segment:
        segments.append(current_segment.strip())

    return segments

#transcript_segments = split_transcript(transcript)

def summarize_segments(transcript_segments, episode_id, summary_type="detailed"):
    """
    Summarizes each transcript segment using GPT-4 and returns structured data.
    """
    dataset = []

    for i, segment in enumerate(transcript_segments):
        prompt = f"Summarize the following transcript in a {summary_type} way:\n\n{segment}"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )

        summary = response["choices"][0]["message"]["content"]

        # Store data in structured format
        dataset.append({
            "segment_id": i + 1,
            "episode_id": episode_id,
            "transcript_segment": segment,
            "summary_segment": summary
        })

    return dataset

#episode_id = 101
#dataset = summarize_segments(transcript_segments, episode_id)
#df = pd.DataFrame(dataset)
#output_path = "podcast_dataset.csv"
#df.to_csv(output_path, index=False)
#print(f"✅ Dataset saved at {output_path}")

def get_embedding(text):
    """
    Generates an embedding for the given text using OpenAI API.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response["data"][0]["embedding"])

def index_transcript(csv_path="podcast_dataset.csv"):
    """
    Converts transcript segments into embeddings and stores in FAISS.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    segments = df["transcript_segment"].tolist()
    
    # Create FAISS index
    d = 1536  # OpenAI embeddings have 1536 dimensions
    index = faiss.IndexFlatL2(d)  # L2 Distance-based FAISS index

    embeddings = []
    for segment in segments:
        embedding = get_embedding(segment)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, "podcast_index.faiss")
    df.to_csv("indexed_podcast_data.csv", index=False)  # Save indexed data for reference

    #print("✅ Transcript embedded & indexed in FAISS.")

#idx_t = index_transcript()

#index = faiss.read_index("podcast_index.faiss")
#df = pd.read_csv("indexed_podcast_data.csv")

def query_podcast(question, index, df):
    """
    Finds the most relevant podcast segment for a user's query.
    """
    question_embedding = np.array(openai.Embedding.create(
        model="text-embedding-ada-002",
        input=question
    )["data"][0]["embedding"]).astype('float32')
    
    # Search FAISS index for the closest match
    D, I = index.search(np.array([question_embedding]), k=1)  # Get top 1 result
    
    return df.iloc[I[0][0]]["transcript_segment"]  # Return the closest transcript


#user_question = "Does AI have a future?"
#answer = query_podcast(user_question)

def generate_answer(question, index, df):
    """
    Uses GPT-4 to generate an answer based on the retrieved podcast segment.
    """
    relevant_segment = query_podcast(question, index, df)
    
    prompt = f"""
    You are an AI assistant answering questions strictly based on the retrieved transcript below. You must prioritize using only the information present in the transcript. However, if the transcript partially answers the user's query but lacks context needed to make the response meaningful, you may generate only the minimum necessary external information to bridge the gap. Any generated information must be directly relevant and should not introduce speculative or unrelated details.
    
    Podcast Transcript Segment:
    "{relevant_segment}"
    
    Question: {question}
    
    Answer:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[{"role": "system", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

#answer = generate_answer(user_question)
#print("\n🤖 Chatbot Response:\n", answer)

def generate_speech(response_text, output_filename):
    """
    Converts text response into speech using Googles TTS API.
    """
    try:
        tts = gTTS(text=response_text, lang='en')
        tts.save(output_filename)
        #print(f"✅ Speech generated and saved as {output_filename}")

    except Exception as e:
        print(f"❌ Error generating speech: {e}")

#generate_speech(answer, "chatbot_response.mp3")

st.title("🎙️ AI Podcast Summarizer & Query Bot")
openai.api_key = os.environ.get("OPENAI_API_KEY")
uploaded_file = st.file_uploader("Upload a podcast audio file (MP3)", type=["mp3"])
if uploaded_file is not None:
    with open("uploaded_podcast.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Podcast uploaded successfully! 🎉")
    chunk_files = split_audio_by_size("uploaded_podcast.mp3")
    final_transcript = transcribe_audio_chunks(chunk_files)
    #st.write(final_transcript)
    transcript = clean_transcript(final_transcript)
    transcript_segments = split_transcript(transcript)
    episode_id = 101
    dataset = summarize_segments(transcript_segments, episode_id)
    df = pd.DataFrame(dataset)
    output_path = "podcast_dataset.csv"
    df.to_csv(output_path, index=False)
    df = pd.read_csv(output_path)
    segments = df["transcript_segment"].tolist()
    d = 1536  
    index = faiss.IndexFlatL2(d) 
    embeddings = []
    for segment in segments:
        embedding = get_embedding(segment)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    faiss.write_index(index, "podcast_index.faiss")
    df.to_csv("indexed_podcast_data.csv", index=False)
    index = faiss.read_index("podcast_index.faiss")
    df = pd.read_csv("indexed_podcast_data.csv")
    st.success("Podcast is summarized sucessfully! 🎉")
    user_question = st.text_input("Ask a question about the podcast:")
    if user_question:
        answer = generate_answer(user_question, index, df)
        st.write("🤖 Chatbot Response: \n", answer)
        generate_speech(answer, "chatbot_response.mp3")
        st.audio("chatbot_response.mp3", format="audio/mp3")
