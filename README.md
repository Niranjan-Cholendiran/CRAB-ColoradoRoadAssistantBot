---
title: CRAB
emoji: 🦀
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.32.2
app_file: app.py
pinned: false
---

# CRAB- Colorado Road Assistant Bot  

<div style="text-align:center;">
    <img src="https://github.com/Niranjan-Cholendiran/CRAB-ColoradoRoadAssistantBot/assets/78549555/f0cd3395-2997-4501-9735-0e6c405afd29" alt="Alt text" width= 200px>
</div>


CRAB is a chat assistant designed to tackle any queries straight from Colorado’s driver handbook. Powered by cutting-edge technology, CRAB utilizes Google’s Gemini language model, Pinecone vector database, and is wrapped with LangChain to provide specific and accurate answers to your road rule questions.

Find more about CRAB in this [LinkedIn post](www.linkedin.com) and see CRAB in action [here](https://huggingface.co/spaces/NiranjanC/CRAB-ColoradoRoadAssistantBot).

## Installation

1. **Install Python Packages:**

Use the following command to install all the required Python packages:

```bash
pip install -r requirements.txt
```

2. **Set Up Environment Variables:**

Create a `.env` file in the root directory and add the following secret codes:

```plaintext
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
PINECONE_HOST=your_pinecone_host
```

3. **Create Pinecone Index:**

Execute the `PineconeDataPrep.ipynb` notebook to create the Pinecone index.

4. **Run the application**

Run the Streamlit application:
```bash
streamlit run app.py
```




