"""Evidence-grounded generative AI for research-paper citation.

Pipeline: PDF -> chunks + metadata -> vector DB + BM25 -> hybrid retrieval
-> passage-first generation -> per-claim verification -> labelled, cited answer.
"""
__version__ = "0.1.0"
