

"""
Evals done in the paper:

1. Performance: 
GPT-2 as normal vs SAE inserted ModuleNotFoundError

CE Loss Increase = CE(SAE Model) - CE(Base Model) 
CE = cross entropy loss

small increase = sae preserves models behavior better

2. Sparsity and Feature Usage:
    1. L0 per data FloatingPointError - lower is better, perf from fewer features
    2. Number of alive features - fewer features = more efficient dictinoary
    3. l1 (sparsity penalty) - actual loss



3. Downstream reconstruction:

for e2e+downstream SAE: penalize mismatch between later layers

||a^(k) - ahat^(k)||^2_2 over all downstream leayer k > lambda



4. Geometry of dictinoary (feature structure):

Look at Dictionary Matrix (rows = features, cols = hidden dims)

    1. within dictionary nearest neighbor cosine (high similiary = more feature splitting and lots of nearly parallel features)
    2. cross seed similarity (train same ase on diff seeds, high similarity = stable features across seeds)
    3. Cross-type similarity (compare dictionaries)

"""


