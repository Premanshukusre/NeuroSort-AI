import time
from transformers import pipeline

print("Loading pipeline...")
start = time.time()
classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
print(f"Loaded in {time.time() - start:.2f}s")

text = "Title: Advanced Computer Science Lab Practical 04.\nObjective: Configure a BGP routing table for the simulated network.\nConclusion: Network was configured successfully."
labels = ['Government Bill', 'Education and Academics', 'Legal Case', 'Technology', 'Healthcare', 'Finance', 'Administrative Document']

start = time.time()
out = classifier(text, labels, hypothesis_template="The topic of this document is {}.")
print(f"\nInference in {time.time() - start:.2f}s")
print(f"Top Category: {out['labels'][0]} ({out['scores'][0]*100:.2f}%)")
