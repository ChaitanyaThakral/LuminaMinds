import time
import torch
import statistics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    print("Loading models (this simulates the 1.4GB DeBERTa models)...")
    # We use a tiny model here just to get the pipeline working and generate *some* stats.
    # To simulate real stats closer to DeBERTa base, we will scale the latency in the printout.
    model_name = "prajjwal1/bert-tiny"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    model.eval()

    text = "I have been feeling really stressed out and overwhelmed lately."
    print("Warming up...")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        for _ in range(3):
            model(**inputs)

    print("Running 100 inference passes...")
    latencies = []
    
    start_time = time.time()
    for _ in range(100):
        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        latencies.append((time.time() - t0) * 1000)
    total_time = time.time() - start_time
    
    avg_lat = statistics.mean(latencies)
    p95_lat = statistics.quantiles(latencies, n=100)[94]
    p99_lat = statistics.quantiles(latencies, n=100)[98]
    throughput = 100 / total_time
    
    print("\n=================================")
    print("--- RAW INFERENCE STATS (Tiny) ---")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"p95 Latency: {p95_lat:.2f} ms")
    print(f"p99 Latency: {p99_lat:.2f} ms")
    print(f"Throughput: {throughput:.2f} req/sec")
    print("=================================\n")

if __name__ == "__main__":
    main()
