from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the user's question based on the below context:{context}
This is the question:{question}
"""


# Prepare the DB.
embedding_function = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model = "local-model",
          messages=[
                {"role": "user", "content": str(prompt)}
            ],
        temperature = 0.7,
    )
    return response.choices[0].message.content




def main():
    
    history = [
        {"role": "assistant", "content": ""},
    ]
        
    while True:
        query = input("Question: ")

        if query in ["quit","exit","bye","stop"]:
            break
        
        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query, k=4)  
        context_text = " ".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)        

        # history.clear()

        history.append({"role": "user", "content": prompt})

        response = chat_with_gpt(history)
        print("Response: ", response)


if __name__ == "__main__":
    main()
