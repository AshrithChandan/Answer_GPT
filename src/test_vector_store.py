import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from modules.vector_store import add_text, query_text

# Add documents
add_text("The capital of France is Paris.")
add_text("Python is a versatile programming language.")
add_text("Apple is a tech company known for the iPhone.")

# Search
results = query_text("What is the capital of France?")
print("Results:\n")
for res in results:
    print("-", res)
