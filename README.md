# Voicebot with Waves API
![cover]()
## Overview

The Voicebot application is designed to convert text and PDF content into high-quality AI-generated speech using the Waves API. By leveraging the power of LangChain and various NLP techniques, this application allows users to interactively upload documents or input text to receive audio outputs in different voice profiles.

## Features

- **Text Input**: Directly enter text for conversion to speech.
- **PDF Upload**: Upload `.pdf` files to extract text and convert it to speech.
- **Query Processing**: Provide additional commands to summarize or explain the content.
- **Voice Selection**: Choose from a range of voice profiles for audio generation.
- **Streamlit UI**: A simple and user-friendly interface built with Streamlit.

## Technologies Used

- **Streamlit**: For creating the web application interface.
- **LangChain**: For document retrieval and language model interactions.
- **Waves API**: To generate high-quality speech from text.
- **Python**: The core programming language used for development.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voicebot.git
   cd voicebot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary API keys for the Waves API.

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Enter your text, upload a PDF file, or input any additional commands in the provided fields.

4. Select your desired voice profile and click the "Submit" button to generate the audio.

5. Listen to the generated audio directly in the browser.

## API Integration

This application integrates with the Waves API for speech generation. You will need to obtain an API key from the Waves platform and input it into the application. 


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Waves API](https://waves-api.smallest.ai/)
