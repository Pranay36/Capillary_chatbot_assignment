# Capillary Chatbot🤖

Capillary chatbot using Langchain, Google Gemini Pro &amp; FAISS Vector DB with Seamless Streamlit Deployment. Get instant, Accurate responses from Awesome Google Gemini OpenSource language Model.

## 📝 Description
The Capillary Chat Bot is a Streamlit-based web application designed to facilitate interactive conversations with a chatbot. The app allows users tinteract with Capillary knowledge base, extract text information from them, and train a chatbot using this extracted content. Users can then engage in real-time conversations with the chatbot.



## 🎯 How It Works:
------------


The application follows these steps to provide responses to your questions:

1. **Scrap Data from Capillary docs** : the knowledgfe based is created using capillary docs consist of total 6 documents, details from each document is stored in json format.

2. **Text Chunking** : The extracted text is divided into smaller chunks that can be processed effectively totaol of 7340 chunck got created.

3. **Language Model** : The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. **Similarity Matching** : When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. **Response Generation** : The selected chunks are passed to the language model, which generates a response based on the relevant content of the Knowlwdge base.


## 🌟Requirements

- **Streamlit** : A Python library for building web applications with interactive elements.
- **google-generativeai** : It is a package that provides generative AI capabilities for chatbots and virtual agents. It can be used in a variety of applications such as content generation, dialogue agents, summarization and classification systems and more.
- **python-dotenv** : A library for loading environment variables from a `.env` file. This is commonly used to store configuration settings, API keys, and other sensitive information outside of your code.
- **langchain** : A custom library for natural language processing tasks, including conversational retrieval, text splitting, embeddings, vector stores, chat models, and memory.
- **beautifulsoup4** : A Python library used to parse HTML and XML documents, ideal for web scraping and extracting data from webpages..
- **faiss-cpu** : FAISS (Facebook AI Similarity Search) is a library developed by Facebook for efficient similarity search, Machine Learning Embeddings,Information Retrieval, content-based filtering and clustering of dense vectors.
- **requests** : A user-friendly HTTP library for Python that lets you send HTTP/1.1 requests (like GET, POST) easily and access web content.
- **langchain_google_genai** : It is a package that provides an integration between LangChain and Google’s generative-ai SDK. It contains classes that extend the Embeddings class and provide methods for generating embeddings. The package can be used in a multipdf chatbot application to extract textual data from PDF documents and generate Accurate responses to user queries.
