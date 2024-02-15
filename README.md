# RAG APPLICATION POWERED BY GOOGLE SEARCH

This application uses the Google Custom Search API to search for information on a given query.

This Google Custom Search API returns a list of relevant website. 

I used the data from the website and converted into text using langchain WebBaseLoader. This data is then divided into chunks and used embeddeing to convert it into vector format using HuggingFaceInstructEmbeddings

Then we perform vector search using the input query and get the relevant result from the vector store and this is given to the llm to give reponse 


## Advantage 
It gives the user up to date infomation since the llm model doesnt know the latest information 

## Disadvantage 
If you process too many websites then it will take time to process that info 

## How to use it 
### Create a .env file
```
GOOGLE_API_KEY="XXXXXXXXXXXXXXXX"
GOOGLE_CSE_ID="XXXXXXXXXXXXXXXXX"
```

### Pip install requirements
```
pip install -r requirements.txt
```

### Run the application 
```
streamlit run chatwithWebsite.py
```