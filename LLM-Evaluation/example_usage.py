import llm_evaluation
import numpy as np

# Simulate classification outputs from 4 LLMs on 10 samples
np.random.seed(0)
simulated_outputs = np.random.randint(1, 6, size=(4, 10))  # values in {1,2,3,4,5}

D, D_tilde, best_llm = llm_evaluation.llm_consensus_evaluation(simulated_outputs)

print("Raw Deviation Scores (D):", D)
print("Normalized Deviation Scores (D_tilde):", D_tilde)
print("Most aligned LLM index:", best_llm)
