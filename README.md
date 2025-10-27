# Lab 4 Performance Comparison

| Model              | Total | correct | incorrect | Accuracy |
|--------------------|-------|---------|-----------|----------|
| Our implementation | 3000  | 2319    | 681       | 77.3% (0.773)    |
| scikit-learn*   | 3000  | 2772    | 228       | 92.4% (0.924)    |

*Scikit-learn is a real machine learning library so it meets the requirement And also in codespaces, we can install it with one command so it is quick and compatible.

**Evaluation:**  
Our Python decision tree got 77.3% accuracy and the scikit-learn decision tree got 92.4% accuracy.
The library model performed better than ours. Likely reasons are different tie breaking when information gain is equal, handling of unseen values in test data, and small implementation details like split selection order, numeric threshold choices.
