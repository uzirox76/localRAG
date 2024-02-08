## This is an example of a local RAG.

- It creates a persistent ChromaDB with embeddings (using HuggingFace model) of all the PDFs in ./data/
- Then you can query the db with 2 files: one's using simple prompt, and one (the "streaming" one) with Streamlit in a website (hosted locally).
- The results are from a local LLM model hosted with LM Studio or others methods.
- There's even an app to update the documents in the ChromaDB database.

Based on work from https://github.com/alejandro-ao/ask-multiple-pdfs and https://github.com/samwit/langchain-tutorials

Have fun. 

_Uzirox_
