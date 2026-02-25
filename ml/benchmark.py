import time
import requests
import statistics
import subprocess
import threading
import sys

def run_server():
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def main():
    print("Starting ML server...")
    server = run_server()
    time.sleep(10) # wait for models to load

    print("Running API Benchmark...")
    url = "http://127.0.0.1:8000/predict"
    payload = {"text": "I have been feeling really stressed out and overwhelmed lately."}
    
    latencies = []
    success = 0
    errors = 0
    
    # Warmup
    for _ in range(3):
        try:
            requests.post(url, json=payload)
        except:
            pass

    print("Measuring latency for 50 requests...")
    start_time = time.time()
    for _ in range(50):
        req_start = time.time()
        try:
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                success += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
        latencies.append((time.time() - req_start) * 1000)
    
    total_time = time.time() - start_time
    
    if success == 0:
        print("All requests failed!")
        server.terminate()
        return

    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=100)[94] if len(latencies) > 1 else latencies[0]
    p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
    throughput = success / total_time
    
    print("\n--- ML Inference API Stats ---")
    print(f"Total Requests: {success + errors}")
    print(f"Success Rate: {(success / (success + errors)) * 100:.2f}%")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"p95 Latency: {p95_latency:.2f} ms")
    print(f"p99 Latency: {p99_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} req/sec")
    
    server.terminate()

if __name__ == "__main__":
    main()
