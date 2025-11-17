import argparse
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans

# -------------------------------------------------------
# Load model
# -------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------------
# Load JSON
# -------------------------------------------------------
def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

# -------------------------------------------------------
# Normalize to list of articles
# -------------------------------------------------------

from dateutil import parser
from datetime import timezone

def parse_date(article):
    # Extract any possible date field
    raw_date = (
        article.get("date")
        or article.get("published_at")
        or article.get("created_at")
        or ""
    )

    # If no date or invalid value → fallback
    if not raw_date or raw_date in ["unknown", "null", "None", "-", ""]:
        return parser.parse("1970-01-01T00:00:00Z")

    try:
        dt = parser.parse(raw_date)

        # If datetime is naive → assign UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:
        # If parsing fails → fallback
        return parser.parse("1970-01-01T00:00:00Z")


def normalize_data(data):
    # dataset is {"items": [...]}
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    # already list
    if isinstance(data, list):
        return data
    return [data]

# -------------------------------------------------------
# Keep only source_rating > 8
# -------------------------------------------------------
def filter_by_rating(data):

    return [d for d in data if d.get("source_rating", 0) > 8]

# -------------------------------------------------------
# Semantic retrieval using title + story
# -------------------------------------------------------
def get_relevant_articles(data, topic, top_k=30):
    topic_emb = model.encode(topic, convert_to_tensor=True)

    texts = [(d.get("title", "") + " " + d.get("story", "")) for d in data]
    embeddings = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(topic_emb, embeddings)[0]
    top_results = scores.topk(k=min(top_k, len(scores)))

    return [data[int(i)] for i in top_results.indices]

# -------------------------------------------------------
# Narrative summary
# -------------------------------------------------------
def build_narrative_summary(articles, topic):
    if not articles:
        return f"No relevant articles found for topic '{topic}'."

    summary_sentences = []
    for a in articles[:8]:
        title = a.get("title", "Untitled")
        src = a.get("source", "Unknown source")
        summary_sentences.append(f"{title} — reported by {src}.")

    return " ".join(summary_sentences)

# -------------------------------------------------------
# Timeline
# -------------------------------------------------------
def build_timeline(articles):
    sorted_articles = sorted(articles, key=lambda a: a.get("parsed_date"))

    timeline = []
    for a in sorted_articles:
        timeline.append({
            "date": a.get("published_at"),
            "headline": a.get("title"),
            "url": a.get("url"),
            "why_it_matters": a.get("story", "")[:200]
        })
    return timeline


# -------------------------------------------------------
# Clusters
# -------------------------------------------------------
def build_clusters(articles, num_clusters=4):
    if len(articles) < 2:
        return {0: articles}

    texts = [a.get("title", "") for a in articles]
    embeddings = model.encode(texts)

    if len(embeddings) < num_clusters:
        num_clusters = len(embeddings)

    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(articles[idx])

    return clusters

# -------------------------------------------------------
# Narrative graph
# -------------------------------------------------------
def build_graph(articles):
    graph = []
    for i, a1 in enumerate(articles):
        for j, a2 in enumerate(articles):
            if i == j:
                continue

            title1 = a1.get("title", "")
            title2 = a2.get("title", "")

            score = util.cos_sim(
                model.encode(title1),
                model.encode(title2)
            ).item()

            if score > 0.60:
                relation = "builds_on"
            elif score > 0.45:
                relation = "adds_context"
            else:
                relation = "contradicts"

            graph.append({
                "from": i,
                "to": j,
                "relation": relation,
                "score": score
            })

    return graph

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--data", default="/content/14e9e4cc-9174-48da-ad02-abb1330b48fe.json")
    parser.add_argument("--output", default="/content/narrative_output.json")
    args = parser.parse_args()

    # Load JSON
    raw = load_data(args.data)

    # Normalize list format
    raw = normalize_data(raw)

    # Parse dates
    for article in raw:
        article["parsed_date"] = parse_date(article)

    # Filter
    filtered = filter_by_rating(raw)

    # Semantically relevant
    relevant = get_relevant_articles(filtered, args.topic)

    # Clustering
    clusters = build_clusters(relevant)

    output = {
        "narrative_summary": build_narrative_summary(relevant, args.topic),
        "timeline": build_timeline(relevant),
        "clusters": {
            str(k): [{"title": art.get("title"), "url": art.get("url")} for art in v]
            for k, v in clusters.items()
        },
        "graph": build_graph(relevant)
    }

    # ✨ Write to JSON file
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Final narrative JSON written to: {args.output}")

if __name__ == "__main__":
  main()

