from qdrant_client import QdrantClient
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

import re


def initialize_vector_store(url, grpc_port, collection_name, embed_model):
    client = QdrantClient(url, grpc_port=grpc_port, prefer_grpc=True)
    qdrant = Qdrant(client, collection_name=collection_name, embeddings=embed_model)

    return qdrant


def store_vectorized_textual_dfg(file_content, filename, embeds_model, address, port, collection_name):
    if filename.endswith('.txt'):
        source = " ".join(filename.strip(".txt").split("_")[:-2]).capitalize()
        title = " ".join(filename.strip(".txt").split("_")[-2:]).capitalize()
    else:
        source = filename.split(".")[0].capitalize()
        title = f"{source} Process Model"

    metadata = {'text': file_content, 'source': source, 'title': title}

    qdrant_store = Qdrant.from_texts(
        [file_content],
        embeds_model,
        metadatas=[metadata],
        url=address,
        prefer_grpc=True,
        grpc_port=port,
        collection_name=collection_name,
    )

    return qdrant_store


def store_fixed_chunked_text(file_content, filename, embeds_model, address, port, collection_name):
    source = filename.strip('.txt').strip('.bpmn').capitalize()
    title = f"{source} Model chunk"
    batch_size = 128
    qdrant_store = 'No qDrant store.'
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=batch_size, chunk_overlap=10)
    splits = text_splitter.split_text(file_content)
    all_splits = text_splitter.create_documents(splits)

    for document in all_splits:
        chunk = document.page_content
        metadata = {'text': chunk, 'source': source, 'title': title}
        qdrant_store = Qdrant.from_texts(
            [chunk],
            embeds_model,
            metadatas=[metadata],
            url=address,
            prefer_grpc=True,
            grpc_port=port,
            collection_name=collection_name,
        )
    return qdrant_store


def store_recursive_chunked_text(file_content, filename, embeds_model, address, port, collection_name):
    source = filename.strip('.txt').strip('.bpmn').capitalize()
    title = f"{source} Model chunk"
    batch_size = 128
    qdrant_store = 'No qDrant store.'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=batch_size, chunk_overlap=10)
    splits = text_splitter.split_text(file_content)
    all_splits = text_splitter.create_documents(splits)

    for document in all_splits:
        chunk = document.page_content
        metadata = {'text': chunk, 'source': source, 'title': title}
        qdrant_store = Qdrant.from_texts(
            [chunk],
            embeds_model,
            metadatas=[metadata],
            url=address,
            prefer_grpc=True,
            grpc_port=port,
            collection_name=collection_name,
        )
    return qdrant_store


def store_vectorized_bpmn(bpmn_model, filename, embeds_model, address, port, collection_name):
    source = filename.strip(".bpmn").capitalize()
    title = "BPMN Model"
    metadata = {'text': bpmn_model, 'source': source, 'title': title}

    qdrant_store = Qdrant.from_texts(
        [bpmn_model],
        embeds_model,
        metadatas=[metadata],
        url=address,
        prefer_grpc=True,
        grpc_port=port,
        collection_name=collection_name,
    )

    return qdrant_store


def store_vectorized_chunked_bpmn(bpmn_model, filename, embeds_model, address, port, collection_name):
    source = filename.strip(".bpmn").capitalize()
    title = f"{source} BPMN Model chunk"
    batch_size = 32
    qdrant_store = 'No qDrant store.'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=batch_size, chunk_overlap=15)
    all_splits = text_splitter.split_documents(bpmn_model)

    for document in all_splits:
        chunk = document.page_content
        metadata = {'text': chunk, 'source': source, 'title': title}
        qdrant_store = Qdrant.from_texts(
            [chunk],
            embeds_model,
            metadatas=[metadata],
            url=address,
            prefer_grpc=True,
            grpc_port=port,
            collection_name=collection_name,
        )

    return qdrant_store


def store_vectorized_semantically_chunked_bpmn(chunks, filename, embeds_model, address, port, collection_name):
    source = filename.strip(".bpmn").capitalize()
    title = f"{source} BPMN Model chunk"
    qdrant_store = 'No qDrant store.'

    for chunk in chunks:
        metadata = {'text': chunk, 'source': source, 'title': title}
        qdrant_store = Qdrant.from_texts(
            [chunk],
            embeds_model,
            metadatas=[metadata],
            url=address,
            prefer_grpc=True,
            grpc_port=port,
            collection_name=collection_name,
        )

    return qdrant_store


def store_vectorized_text(file_content, filename, embeds_model, address, port, collection_name):
    source = filename.removesuffix("_output.txt").capitalize()
    title = f'{source} process parameters'
    metadata = {'text': file_content, 'source': source, 'title': title}

    qdrant_store = Qdrant.from_texts(
        [file_content],
        embeds_model,
        metadatas=[metadata],
        url=address,
        prefer_grpc=True,
        grpc_port=port,
        collection_name=collection_name,
    )

    return qdrant_store


def parse_bpmn_in_chunks(content):
    """
    Parses the given BPMN content and returns a list of semantic chunks, one for each element.

    Parameters:
    - content (str): The BPMN content to be parsed.

    Returns:
    - semantic_chunks (list): A list of semantic chunks extracted from the BPMN content.

    Example:
    >>> bpmn_content = '<process>...</process>'
    >>> parse_bpmn_in_chunks(bpmn_content)
    ['<lane>...</lane>', '<task>...</task>', '<endEvent>...</endEvent>', ...]
    """

    semantic_chunks = []
    process_match = re.search(r'<process.*?</process>', content, re.DOTALL)
    if process_match:
        process_content = process_match.group(0)
    else:
        return []

    element_patterns = [
        r'<lane.*?</lane>',
        r'<task.*?</task>',
        r'<endEvent.*?</endEvent>',
        r'<startEvent.*?</startEvent>',
        r'<receiveTask.*?</receiveTask>',
        r'<exclusiveGateway.*?</exclusiveGateway>',
        r'<sendTask.*?</sendTask>',
        r'<parallelGateway.*?</parallelGateway>',
        r'<textAnnotation.*?</textAnnotation>',
        r'<sequenceFlow.*?/>',
        r'<association.*?/>'
    ]
    for pattern in element_patterns:
        semantic_chunks.extend(re.findall(pattern, process_content, re.DOTALL))

    return semantic_chunks


def retrieve_context(vector_index, query, num_chunks):
    retrieved = vector_index.similarity_search(query, num_chunks)
    retrieved_text = ''
    for i in range(len(retrieved)):
        content = retrieved[i].page_content
        if i != len(retrieved) - 1:
            retrieved_text += f'{content}\n\n'
        else:
            retrieved_text += f'{content}'

    return retrieved_text


def delete_qdrant_collection(q_url, q_grpc_port, q_collection_name):
    qdrant_client = QdrantClient(url=q_url, grpc_port=q_grpc_port, prefer_grpc=True)
    qdrant_client.delete_collection(q_collection_name)
    qdrant_client.close()
