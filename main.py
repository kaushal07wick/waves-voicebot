import requests
import uuid
import os
import wave
import streamlit as st 
import re
import shutil
# LangChain-related imports for document retrieval and AI agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

ICON_MAIN = "images/s.png"
MAIN_IMAGE = "images/s.png"
options = [ICON_MAIN, MAIN_IMAGE]
st.logo(ICON_MAIN, size="medium", icon_image=MAIN_IMAGE)

st.set_page_config(page_title="Voicebot")
st.title('Voicebot for Documents')
st.image("images/cover.png")
st.caption("Generate Hyper-Realistic Speech for your Document with Low Latency Inference. See the pricing at [Waves API](https://waves.smallest.ai/)")

AUDIO_DIR = "audio_output"
SAMPLE_RATE = 16000  # The sample rate in Hz
NUM_CHANNELS = 1  # Assuming mono audio, adjust if needed
SAMPLE_WIDTH = 2  # 2 bytes per sample (16-bit PCM)
CHROMA_PATH = "Waves"

# Create directory if it doesn't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

url = "https://waves-api.smallest.ai/api/v1/lightning/get_speech"

# Embedding model for document retrieval
model_name = "BAAI/bge-small-en"
hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

# Load PDF, split into chunks, and store in Chroma
def load_pdf_and_initialize_db(pdf_path, db_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    st.info(pages[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    db_chroma = Chroma.from_documents(chunks, hf, persist_directory=db_path)
    return db_chroma

def clean_text(text):
    # Remove non-printable characters (including control characters)
    cleaned_text = ''.join(c for c in text if c.isprintable())
    
    # Optionally, remove any unwanted characters like extra newlines, tabs, etc.
    cleaned_text = re.sub(r'[\n\t]+', ' ', cleaned_text).strip()
    
    return cleaned_text


def rag_from_pdf(pdf_path, db_path, query_str):
    data = load_pdf_and_initialize_db(pdf_path, db_path)
    docs_chroma = data.similarity_search_with_score(query_str, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
    context_text = clean_text(context_text)
    st.info(context_text)
    docs_chroma.__delitem__(0)

    PROMPT_TEMPLATE = """
    Using the context provided below, use the whole text, keep the sequence of the text proper.
    Ensure that the response is amazing, but short and to the point. Give proper punctuations.
    the context is as follows:
    {context}
    
    Please respond to the following prompt based on the above context: {question}
    Focus on creating an engaging short story that flows naturally.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_str)

    llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation", max_new_tokens=1024, do_sample=False, repetition_penalty=1.03)
    chat_model = ChatHuggingFace(llm=llm)
    response_text = chat_model.predict(prompt)

    return str(response_text).strip()

def main():
    api_key = st.sidebar.text_input("Input the Waves API", type="password")
    text = st.text_input("Enter Any text you want to listen..")
    files = st.file_uploader("Submit **.txt/.pdf** files only", type=['txt', 'pdf'])
    query = st.text_input("Please enter the query; such as (summarize the pdf) or (explain this like I am 5)")
    submit_button = st.button("Submit")

    voice_id = st.sidebar.radio(
        label="**Select Any Voice Id**",
        options=[
            'Emily',
            'Jasmine',
            'Arman',
            'James',
            'Mithali',
            'Aravind',
            'Raj',
        ]
    )


    if submit_button:
        if not api_key:
            st.error("Please provide an API key")
            return

        try:
            payload = None

            # Handle direct text input
            if text and not files:
                st.info("Processing direct text input...")
                payload = {
                    "voice_id": voice_id.lower(),
                    "text": text,
                    "sample_rate": SAMPLE_RATE
                }
            
            # Handle file upload
            elif files:
                st.info("Processing uploaded file...")
                temp_pdf_path = f"temp_{uuid.uuid4()}.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(files.read())

                if query:
                    text_from_rag = rag_from_pdf(temp_pdf_path, CHROMA_PATH, query)
                    st.info("File processing completed")

                    # Check if the response from rag is valid
                    if not text_from_rag:
                        st.error("No text generated from the PDF.")
                        return

                    payload = {
                        "voice_id": voice_id.lower(),
                        "text": text_from_rag,
                        "sample_rate": SAMPLE_RATE
                    }
                else:
                    st.error("Please provide a query for the uploaded PDF.")
                    return

                os.remove(temp_pdf_path)  # Clean up the temporary file
            
            else:
                st.error("Please either enter text or upload a file.")
                return

            # Debugging: Print payload to check its content
            st.write("Payload being sent:", payload)  # Print payload for debugging

            if payload:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
               
                with st.spinner("Generating audio..."):
                    response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    pcm_data = response.content
                    audio_file = f"{uuid.uuid4()}.wav"
                    audio_path = os.path.join(AUDIO_DIR, audio_file)
                    
                    with wave.open(audio_path, 'wb') as wav_file:
                        wav_file.setnchannels(NUM_CHANNELS)
                        wav_file.setsampwidth(SAMPLE_WIDTH)
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes(pcm_data)
                    
                    st.success("Audio generated successfully!")
                    st.audio(audio_path)
                else:
                    st.error(f"API Error: {response.status_code}, {response.text}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during the request: {str(e)}")
        except ValueError as e:
            st.error(str(e))  # Catch the ValueError raised in rag_from_pdf
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
