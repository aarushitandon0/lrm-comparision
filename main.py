import os
from dotenv import load_dotenv, find_dotenv
from base_llm import BaseLLM
from reasoning.linear import LinearReasoning
from reasoning.self_consistent import SelfConsistentReasoning
from reasoning.tree import TreeReasoning
from reasoning.graph import GraphReasoning
from reasoning.mcts import MCTSReasoning
from reasoning.hybrid import HybridTreeConsistentReasoning
from eval.questions import QUESTIONS

load_dotenv(find_dotenv())
API_KEY = os.getenv("GROQ_API_KEY")


def run_layer(name: str, layer, question: str):
    print(f"\n{'='*60}")
    print(f"  Layer : {name}")
    print(f"  Q     : {question[:80]}...")
    print(f"{'='*60}")

    result = layer.solve(question)

    print("\nSteps:")
    for i, step in enumerate(result["steps"], 1):
        text = step[:130] + ("..." if len(step) > 130 else "")
        print(f"  {i}. {text}")

    print(f"\nFinal Answer : {result['answer']}")
    print(f"Stats        : {result['stats']}")


def main():
    llm = BaseLLM(api_key=API_KEY)

    layers = {
        "Linear": LinearReasoning(llm),
        "Self-Consistent": SelfConsistentReasoning(llm),
        "Tree": TreeReasoning(llm),
        "Graph": GraphReasoning(llm),
        "MCTS": MCTSReasoning(llm, num_iterations=8),
        "Hybrid": HybridTreeConsistentReasoning(llm, n_chains=3),
    }

    # Change index to test different questions (0-54)
    question = QUESTIONS[0]["question"]

    for name, layer in layers.items():
        run_layer(name, layer, question)


if __name__ == "__main__":
    main()
