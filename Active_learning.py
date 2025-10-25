import argparse
import json
from pathlib import Path
import re

import faiss
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def parse_args():
    parser = argparse.ArgumentParser(description="Select informative samples using entropy + diversity.")
    parser.add_argument("--dump_dir", required=True, help="Directory containing entropy_relation.pt etc.")
    parser.add_argument("--unlabeled_json", required=True, help="JSON file of unlabeled documents.")
    parser.add_argument("--output_prefix", default="al_round",
                        help="Prefix for files written back to dump_dir (e.g. weighted_embedding_<prefix>.pt).")
    parser.add_argument("--top_k", type=int, default=20, help="Number of clusters to retain when ranking.")
    parser.add_argument("--sample_per_group", type=int, default=10,
                        help="How many samples to draw from each selected cluster.")
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for class entropy term.")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Relative weight between relation and entity entropy.")
    parser.add_argument("--ncentroids", type=int, default=20, help="Number of clusters for MiniBatch KMeans.")
    parser.add_argument("--use_weights", action="store_true",
                        help="If set, weight KMeans with entropy to emphasise uncertain points.")
    return parser.parse_args()


def load_tensors(dump_dir: Path):
    entropy_relation = torch.load(dump_dir / "entropy_relation.pt", map_location="cpu").float()
    entropy_entities = torch.load(dump_dir / "entropy_entities.pt", map_location="cpu").float()
    label_prediction = torch.load(dump_dir / "label_prediction.pt", map_location="cpu").float()
    pooler_output = torch.load(dump_dir / "pooler_output.pt", map_location="cpu").float()
    return entropy_relation, entropy_entities, label_prediction, pooler_output


def detokenize(tokens: list[str]) -> str:
    words = []
    for tok in tokens:
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
        else:
            words.append(tok)

    text = " ".join(words)
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    return text


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    entropy_relation, entropy_entities, label_prediction, pooler_output = load_tensors(dump_dir)

    entropy_relation = entropy_relation.reshape(-1)
    entropy_entities = entropy_entities.reshape(-1)
    n_sample = entropy_relation.shape[0]
    entropy = entropy_relation + args.gamma * entropy_entities
    entropy_np = entropy.cpu().numpy()
    weights = np.ones(n_sample)

    pooler_np = pooler_output.cpu().numpy()
    feature_dim = pooler_np.shape[-1]

    if args.use_weights:
        kmeans = MiniBatchKMeans(
            n_clusters=args.ncentroids, random_state=0, batch_size=256, n_init=3, max_iter=100
        )
        kmeans.fit(pooler_np, sample_weight=entropy_np)
        index = faiss.IndexFlatL2(feature_dim)
        index.add(kmeans.cluster_centers_.astype("float32"))
        _, assignments = index.search(pooler_np.astype("float32"), 1)
    else:
        index = faiss.IndexFlatL2(feature_dim)
        kmeans = faiss.Clustering(feature_dim, args.ncentroids)
        kmeans.train(pooler_np.astype("float32"), index)
        centroids = faiss.vector_to_array(kmeans.centroids).reshape(args.ncentroids, -1)
        index.add(centroids.astype("float32"))
        _, assignments = index.search(pooler_np.astype("float32"), 1)

    assignments = assignments.flatten()
    scores = []
    cluster_members = []

    for cluster_id in range(args.ncentroids):
        mask = assignments == cluster_id
        cluster_members.append(np.where(mask)[0].tolist())
        if not np.any(mask):
            scores.append(-np.inf)
            continue

        weights[mask] = entropy_np[mask] / (entropy_np[mask].sum() + 1e-12)
        mean_entropy = float(entropy_np[mask].mean())

        cluster_pred = label_prediction[mask]
        class_sum = cluster_pred.sum(dim=0, keepdim=False).numpy()
        if class_sum.sum() == 0:
            class_entropy = 0.0
        else:
            freq = class_sum / (class_sum.sum() + 1e-12)
            class_entropy = float(np.sum(np.abs(freq * np.log(freq + 1e-12))))

        scores.append(mean_entropy + args.beta * class_entropy)

    sample_indices = []
    remaining = n_sample
    for cluster_id in np.argsort(scores)[::-1][:args.top_k]:
        members = cluster_members[cluster_id]
        if not members:
            continue
        ent_values = entropy_np[members]
        sorted_members = [m for _, m in sorted(zip(ent_values, members))]
        to_take = min(len(sorted_members), args.sample_per_group, remaining)
        sample_indices.extend(sorted_members[-to_take:])
        remaining -= to_take
        if remaining <= 0:
            break

    weighted_embedding = weights.reshape(-1, 1) * pooler_np

    prefix = args.output_prefix
    torch.save(torch.tensor(weighted_embedding, dtype=torch.float32),
               dump_dir / f"weighted_embedding_{prefix}.pt")
    torch.save(torch.tensor(assignments, dtype=torch.long),
               dump_dir / f"class_number_{prefix}.pt")
    torch.save(torch.tensor(sample_indices, dtype=torch.long),
               dump_dir / f"selected_indices_{prefix}.pt")

    with open(args.unlabeled_json, "r") as fr:
        unlabeled_docs = json.load(fr)

    sampled_docs = [unlabeled_docs[idx] for idx in sample_indices]
    with open(dump_dir / f"sampling_json_{prefix}.json", "w") as fw:
        json.dump(sampled_docs, fw)

    readable_sentences = []
    for doc in sampled_docs:
        sentence = detokenize(doc.get("tokens", []))
        orig_id = doc.get("orig_id", "")
        readable_sentences.append((orig_id, sentence))

    with open(dump_dir / f"sampling_text_{prefix}.txt", "w") as fw:
        for orig_id, sentence in readable_sentences:
            line = f"{orig_id}\t{sentence}" if orig_id else sentence
            fw.write(line + "\n")

if __name__ == "__main__":
    main()

