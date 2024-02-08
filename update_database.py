from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import uuid

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    aggiorna(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks



def aggiorna(chunks: list[Document]):
    '''
    In the first line, a unique UUID is generated for each document by using the uuid.uuid5() function, which creates a UUID using 
    the SHA-1 hash of a namespace identifier and a name string (in this case, the content of the document).

    The if condition in the list comprehension checks whether the ID of the current document exists in the seen_ids set:

    If it doesn't exist, this implies the document is unique. It gets added to seen_ids using seen_ids.add(id), and the document gets included in unique_docs.
    If it does exist, the document is a duplicate and gets ignored.
    The or True at the end is necessary to always return a truthy value to the if condition, because seen_ids.add(id) returns None (which is falsy) 
    even when an element is successfully added.

    This approach is more practical than generating IDs using URLs or other document metadata, as it directly prevents the addition of 
    duplicate documents based on content rather than relying on metadata or manual checks.
    '''

    docs = chunks
    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
    unique_ids = list(set(ids))

    # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
    seen_ids = set()
    unique_docs = [doc for doc, id in zip(docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")    

    # Add the unique documents to your database
    db = Chroma.from_documents(unique_docs, embeddings, ids=unique_ids, persist_directory=CHROMA_PATH)
    db.persist()
    db = None
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
