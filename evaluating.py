import json
import configparser
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import numpy
import tqdm
import argparse

config = configparser.ConfigParser()
config.read("config.ini")
cloud_api_key = config["secrets"]["api_key"]
cloud_url = config["qdrant"]["cloud_url"]
qdrant_client = QdrantClient(url=cloud_url, api_key=cloud_api_key)

smaller_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
larger_model = TextEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")


def load_qrels(file_path_qrels: str):
    qrels = {}

    with open(file_path_qrels, "r") as file:
        next(file)
        for line in file:
            query_id, doc_id, score = line.strip().split("\t")
            if int(score) > 0:
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(score)

    return qrels

def load_queries(file_path_queries: str):
    queries = {}

    with open(file_path_queries, "r") as file:
        for line in file:
            row = json.loads(line)
            queries[row["_id"]] = {**row}

    return queries

def get_naive(query_smaller: numpy.ndarray, query_larger: numpy.ndarray, collection_name: str = "discovery_agents") -> list[models.ScoredPoint]:
    return qdrant_client.query_points( #do we need to check top-10 by smart model? ig no
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_smaller,
                using="smaller",
                limit=10,
            )
        ],
        query=query_larger,
        using="larger",
        with_payload=True,
        limit=10
    ).points

def get_top_three(query_smaller: numpy.ndarray, collection_name: str = "discovery_agents") -> list[int]:
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_smaller,
        using="smaller",
        with_payload=True,
        limit=3
    ).points

def get_top_three_rescore(query_larger: numpy.ndarray, top_three_scored_ids: list[int], collection_name: str = "discovery_agents") -> list[models.ScoredPoint]:
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_larger,
        using="larger",
        limit=3,
        with_payload=True,
        query_filter=models.Filter(
            must=models.HasIdCondition(
                has_id=top_three_scored_ids
            )
        )
    ).points

def get_discovery(query_smaller: numpy.ndarray, positive_context: int, negative_context: int, ids_to_exclude: list[int], collection_name: str = "discovery_agents") -> list[models.ScoredPoint]:
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=query_smaller,
                context=[
                    models.ContextPair(
                        positive=positive_context,
                        negative=negative_context,
                    )
                ],
            )
        ),
        query_filter=models.Filter(
            must_not=models.HasIdCondition(
                has_id=ids_to_exclude
            )
        ),
        using="smaller",
        with_payload=True,
        limit=3,
        #search_params=models.SearchParams( #for now
        #    exact=True
        #)
    ).points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path-queries", type=str)
    parser.add_argument("--input-path-qrels", type=str)
    parser.add_argument("--collection-name", type=str, default="discovery_agents")
    parser.add_argument("--total-queries-in-dataset", type=int, default=648)

    args = parser.parse_args()
    queries = load_queries(args.input_path_queries)
    qrels = load_qrels(args.input_path_qrels)

    no_reordering = 0
    top_1_hit_naive = 0
    top_1_hit_with_discovery = 0
    queries_amount = 0

    for query_id in tqdm.tqdm(qrels, total=args.total_queries_in_dataset):
        queries_amount += 1

        query = queries[query_id]["text"]
        query_smaller = list(smaller_model.query_embed(query))[0]
        query_larger = list(larger_model.query_embed(query))[0]

        top_ten = get_naive(query_smaller, query_larger, collection_name=args.collection_name)
        if top_ten[0].payload["document_id"] in qrels[query_id]:
            top_1_hit_naive += 1

        top_three = get_top_three(query_smaller, collection_name=args.collection_name)
        pre_discovery = get_top_three_rescore(query_larger, [point.id for point in top_three], collection_name=args.collection_name)

        if (pre_discovery[0].id != top_three[0].id) or (pre_discovery[2].id != top_three[2].id):
            positive_context = pre_discovery[0].id
            negative_context = pre_discovery[2].id #or should we use specifically ones changing order? Not necessarily the last one?
            
            discovery = get_discovery(query_smaller, positive_context, negative_context, [point.id for point in pre_discovery], collection_name=args.collection_name)
            discovery_rescored = get_top_three_rescore(query_larger, [point.id for point in discovery], collection_name=args.collection_name) 
            
            if discovery_rescored[0].score > pre_discovery[0].score: #interested in top-1, RAG
                discovery_result = discovery_rescored[0]
            else:
                discovery_result = pre_discovery[0]
            
            if discovery_result.payload["document_id"] in qrels[query_id]:
                top_1_hit_with_discovery += 1

        else:
            no_reordering += 1
            if pre_discovery[0].payload["document_id"] in qrels[query_id]:
                top_1_hit_with_discovery += 1
        
    print(f"On {no_reordering} queries out of {queries_amount} top-3 rescoring with an \"agent\" didn't change top-1 result")

    print(f"Quality naive (top-10 reordered) on top-1 hits: {top_1_hit_naive / queries_amount}")
    print(f"Quality with discovery on top-1 hits: {top_1_hit_with_discovery / queries_amount}")

if __name__ == "__main__":
    main()
       



