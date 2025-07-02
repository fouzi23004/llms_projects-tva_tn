import os
import pypdf
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def preprocess_text(text):
    """Clean and preprocess text for better chunking."""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove common PDF artifacts
    text = text.replace('\x00', '')  # null characters
    text = text.replace('•', '-')  # bullet points
    # Remove very short lines (likely headers/footers)
    lines = text.split('\n')
    meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 15]
    return ' '.join(meaningful_lines)


def create_contextual_chunks(text, base_chunk_size=400, overlap=80):
    """Create chunks that preserve context better."""
    # Try paragraph-based chunking first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If adding this paragraph exceeds size, save current chunk
        if len(current_chunk) + len(paragraph) + 2 > base_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Keep some overlap for context
                words = current_chunk.split()
                if len(words) > 20:
                    overlap_text = ' '.join(words[-10:])  # Last 10 words
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk = paragraph
        else:
            current_chunk += ("\n\n" + paragraph if current_chunk else paragraph)

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out very short chunks
    return [chunk for chunk in chunks if len(chunk.strip()) > 60]


# Set up the embeddings with better model
MODEL = 'BAAI/bge-small-en-v1.5'  # Better model version
embeddings = SentenceTransformer(MODEL)

# Load PDF documents from the data folder with improved processing
data_folder = "data"
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(data_folder, filename)
        print(f"Processing: {filename}")

        try:
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()

            # Process each page
            for page_num, doc in enumerate(raw_docs):
                # Clean the text
                cleaned_text = preprocess_text(doc.page_content)

                # Skip very short pages
                if len(cleaned_text.strip()) < 100:
                    continue

                # Create contextual chunks
                chunks = create_contextual_chunks(cleaned_text)

                # Create document objects for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    # Create a new document object
                    chunk_doc = type(doc)(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'word_count': len(chunk.split()),
                            'original_page': page_num
                        }
                    )
                    documents.append(chunk_doc)

            print(f"Created {len([d for d in documents if filename in d.metadata['source']])} chunks from {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

print(f"Total documents processed: {len(documents)}")

# Set up Qdrant client
client = QdrantClient("localhost", port=6333)

# Get the actual vector size from the model
vector_size = embeddings.get_sentence_embedding_dimension()

# Create collection in Qdrant database
collection_name = "document_collection"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)  # Start fresh for better results

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "content": VectorParams(size=vector_size, distance=Distance.COSINE)
    }
)


# Improved function to chunk documents and upload to Qdrant
def chunked_metadata(data, client=client, collection_name="document_collection", batch_size=32):
    if not data:
        print("No documents to process")
        return

    total_docs = len(data)
    processed = 0

    for i in range(0, total_docs, batch_size):
        batch = data[i:i + batch_size]
        points_batch = []

        # Prepare content for batch embedding
        content_texts = []
        valid_items = []

        for item in batch:
            content = item.page_content.strip()
            if content and len(content) > 20:  # Only process meaningful content
                content_texts.append(content)
                valid_items.append(item)

        if not content_texts:
            continue

        try:
            print(f"Processing batch {i // batch_size + 1}/{(total_docs - 1) // batch_size + 1}")

            # Generate embeddings in batch for efficiency
            content_vectors = embeddings.encode(
                content_texts,
                batch_size=16,
                show_progress_bar=True,
                normalize_embeddings=True  # Better for cosine similarity
            )

            # Create points for this batch
            for j, item in enumerate(valid_items):
                id = str(uuid4())
                content = item.page_content.strip()

                # Convert numpy array to list for Qdrant
                content_vector = content_vectors[j].tolist()
                vector_dict = {"content": content_vector}

                # Enhanced payload with more metadata
                payload = {
                    "page_content": content,
                    "metadata": {
                        "id": id,
                        "source": item.metadata.get("source", "unknown"),
                        "page": item.metadata.get("page", 0),
                        "chunk_index": item.metadata.get("chunk_index", 0),
                        "word_count": item.metadata.get("word_count", len(content.split())),
                        "char_count": len(content),
                        "original_page": item.metadata.get("original_page", item.metadata.get("page", 0))
                    }
                }

                metadata = PointStruct(id=id, vector=vector_dict, payload=payload)
                points_batch.append(metadata)

            # Upload batch to Qdrant
            if points_batch:
                client.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=points_batch
                )
                processed += len(points_batch)
                print(f"Uploaded: {processed}/{total_docs} documents")

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            continue


# Upload all documents to Qdrant
print("Starting upload to Qdrant...")
chunked_metadata(documents)

# Print information about the collection
try:
    document_collection = client.get_collection("document_collection")
    print(f"\nUpload complete!")
    print(f"Points in collection: {document_collection.points_count}")
    print(f"Collection status: {document_collection.status}")
    print(f"Vector size: {document_collection.config.params.vectors['content'].size}")

    # Test search functionality
    print(f"\nTesting search functionality...")
    test_query = "test query"
    test_vector = embeddings.encode([test_query], normalize_embeddings=True)[0].tolist()

    search_results = client.search(
        collection_name="document_collection",
        query_vector={"name": "content", "vector": test_vector},
        limit=3,
        score_threshold=0.0
    )

    print(f"Search test successful! Found {len(search_results)} results")
    if search_results:
        print(f"Top result score: {search_results[0].score:.4f}")
        print(f"Top result preview: {search_results[0].payload['page_content'][:100]}...")

except Exception as e:
    print(f"Error getting collection info: {e}")