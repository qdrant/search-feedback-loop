import json
import argparse
from itertools import tee
from typing import Iterable, Tuple
import tqdm
import configparser
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

model_small = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_big = TextEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")

def read_beir_dataset(file_path: str) -> Iterable[Tuple[int, str, str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            row = json.loads(line)
            yield idx, row["_id"], f"{row['title']}\n{row['text']}"


def read_points(file_path: str) -> Iterable[models.PointStruct]:
    dataset_iter = read_beir_dataset(file_path)
    
    dataset_for_embedding1, dataset_for_embedding2, dataset_for_points = tee(dataset_iter, 3)
    
    docs_for_larger = (text for _, _, text in dataset_for_embedding1)
    docs_for_smaller = (text for _, _, text in dataset_for_embedding2)
    
    embeddings_larger = model_big.embed(docs_for_larger, batch_size=8, parallel=2)
    embeddings_smaller = model_small.embed(docs_for_smaller, batch_size=8, parallel=2)
    
    for ((idx, doc_id, text), emb_smaller, emb_larger) in zip(dataset_for_points, embeddings_smaller, embeddings_larger):
        yield models.PointStruct(
            id=idx,
            vector={
                "smaller": emb_smaller,
                "larger": emb_larger
            },
            payload={
                "document": text,
                "document_id": doc_id
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--total-points-in-dataset", type=int, default=3600)
    parser.add_argument("--collection-name", type=str, default="discovery_agents")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("config.ini")
    cloud_api_key = config["secrets"]["api_key"]
    cloud_url = config["qdrant"]["cloud_url"]
  
    qdrant_client = QdrantClient(url=cloud_url, api_key=cloud_api_key)

    if not qdrant_client.collection_exists(args.collection_name):
        qdrant_client.create_collection(
            collection_name=args.collection_name,
            vectors_config={
                "smaller": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                ),
                "larger": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                ),
            },
        )

    qdrant_client.upload_points(
        collection_name=args.collection_name,
        points=tqdm.tqdm(
            read_points(args.dataset_path),
            total=args.total_points_in_dataset,
            desc="Uploading points",
        ),
        batch_size=32
    )

if __name__ == "__main__":
    main()
