"""
Generate synthetic training pairs for CodeShield when the PAN corpus is unavailable.

Creates realistic text pairs with controlled similarity levels using text
mutation techniques:
  - Copy (high similarity, label=1)
  - Paraphrase simulation (moderate similarity, label=1)  
  - Topic-shared (low similarity, label=0)
  - Unrelated (very low similarity, label=0)

Output:
  data/pairs.csv
  data/documents/  (synthetic .txt files)
"""
import sys
import os
import random
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DOCS_DIR = Path("data/documents")
OUTPUT_PAIRS_CSV = Path("data/pairs.csv")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Sample base texts covering different academic topics
BASE_TEXTS = {
    "ml_intro": """
Machine learning is a branch of artificial intelligence that enables computers to learn from 
experience without being explicitly programmed. It focuses on developing algorithms that can 
improve automatically through experience. The main categories are supervised learning, where 
models learn from labeled training data, unsupervised learning, where models find patterns in 
unlabeled data, and reinforcement learning, where agents learn through rewards and penalties.
Deep learning, a subset of machine learning, uses neural networks with many layers to model 
complex patterns. Gradient descent is the optimization algorithm most commonly used to train 
these models by minimizing a loss function. Overfitting occurs when a model learns the training 
data too well and fails to generalize to new data. Regularization techniques such as L1 and L2 
penalties, dropout, and early stopping help combat overfitting.
""",

    "climate_change": """
Climate change refers to long-term shifts in global temperatures and weather patterns. While some 
climate variation is natural, scientific evidence overwhelmingly shows that human activities have 
been the main driver of climate change since the 1950s. The burning of fossil fuels releases 
greenhouse gases like carbon dioxide and methane into the atmosphere, trapping heat and causing 
global warming. Effects include rising sea levels, more frequent extreme weather events, melting 
polar ice caps, and disruptions to ecosystems and biodiversity. The Paris Agreement, adopted in 
2015, aims to limit global warming to 1.5 degrees Celsius above pre-industrial levels. Renewable 
energy sources such as solar, wind, and hydropower offer alternatives to fossil fuels that produce 
little to no greenhouse gas emissions.
""",

    "sorting_algorithms": """
Sorting algorithms are fundamental to computer science and arrange data in a specific order. 
Bubble sort repeatedly swaps adjacent elements if they are in the wrong order, with O(n^2) 
average complexity. Merge sort uses divide and conquer, splitting arrays and merging sorted 
halves, achieving O(n log n) in all cases. Quick sort selects a pivot element and partitions 
the array around it, with O(n log n) average but O(n^2) worst case. Heap sort uses a binary 
heap structure to sort in O(n log n) guaranteed time. For nearly sorted arrays, insertion sort 
performs well with O(n) best case. Radix sort, a non-comparative algorithm, sorts integers digit 
by digit and runs in O(nk) where k is the number of digits.
""",

    "photosynthesis": """
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy 
into chemical energy stored as glucose. The overall equation is: 6CO2 + 6H2O + light energy 
→ C6H12O6 + 6O2. This occurs in two stages: the light-dependent reactions in the thylakoid 
membranes, which capture light energy and produce ATP and NADPH while splitting water, and 
the light-independent Calvin cycle in the stroma, which uses this energy to fix carbon dioxide 
into organic compounds. Chlorophyll is the primary pigment that absorbs red and blue light 
while reflecting green, giving plants their color. Factors affecting the rate include light 
intensity, carbon dioxide concentration, temperature, and water availability.
""",

    "french_revolution": """
The French Revolution was a period of radical political and social transformation in France from 
1789 to 1799. It began with the Estates-General and the storming of the Bastille on July 14, 1789. 
The Declaration of the Rights of Man and Citizen established principles of individual liberty and 
equality. The revolution abolished feudalism, the nobility's privileges, and the power of the 
Catholic Church in France. The radical Jacobins under Robespierre led the Reign of Terror, 
executing thousands of perceived enemies. The revolution ended with Napoleon Bonaparte's coup 
in November 1799. It transformed France from an absolute monarchy to a republic and spread 
revolutionary ideals of liberty, equality, and fraternity throughout Europe.
""",

    "neural_networks": """
Neural networks are computational models inspired by the human brain, consisting of layers of 
interconnected nodes called neurons. The input layer receives raw data, hidden layers transform 
it through weighted connections and activation functions, and the output layer produces predictions. 
Backpropagation is the algorithm used to train neural networks by calculating gradients of the 
loss function with respect to each weight and updating them via gradient descent. Common activation 
functions include ReLU, sigmoid, and tanh. Convolutional neural networks excel at image recognition 
by using convolutional filters to detect spatial features. Recurrent neural networks process 
sequential data using hidden states that capture temporal dependencies. Transformer architectures 
use attention mechanisms to model relationships between all positions in a sequence simultaneously.
""",

    "economics_supply_demand": """
Supply and demand is the fundamental economic model describing how prices and quantities of goods 
are determined in a market. The law of demand states that as price increases, quantity demanded 
decreases, all else being equal. The demand curve slopes downward from left to right. The law of 
supply states that as price increases, quantity supplied increases. The equilibrium price is where 
supply equals demand, clearing the market. Shifts in demand can occur due to changes in income, 
tastes, prices of related goods, and consumer expectations. Shifts in supply can result from 
changes in input costs, technology, and the number of producers. Price elasticity measures how 
responsive quantity demanded or supplied is to price changes.
""",

    "cellular_biology": """
Cells are the basic units of life and are classified into prokaryotic and eukaryotic types. 
Prokaryotic cells, such as bacteria, lack a membrane-bound nucleus and organelles. Eukaryotic 
cells, found in animals, plants, and fungi, have a nucleus containing DNA and various organelles. 
The cell membrane is a phospholipid bilayer that regulates what enters and exits the cell. 
Mitochondria are the powerhouses of the cell, generating ATP through cellular respiration. 
The endoplasmic reticulum synthesizes proteins and lipids. The Golgi apparatus processes and 
packages proteins for secretion. Cell division occurs through mitosis for growth and repair, 
and meiosis for sexual reproduction. DNA replication ensures genetic information is accurately 
copied before cell division.
""",
}


def mutate_word_order(text: str, swap_prob: float = 0.15) -> str:
    """Randomly swap some adjacent words."""
    words = text.split()
    for i in range(len(words) - 1):
        if random.random() < swap_prob:
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def paraphrase_text(text: str) -> str:
    """Simulate paraphrasing by substituting some words and reordering sentences."""
    replacements = {
        "is": "represents",
        "uses": "employs",
        "shows": "demonstrates",
        "occurs": "happens",
        "allows": "enables",
        "called": "known as",
        "helps": "assists",
        "include": "encompass",
        "common": "typical",
        "main": "primary",
        "important": "crucial",
        "produces": "generates",
        "contains": "includes",
    }
    for old, new in replacements.items():
        text = text.replace(f" {old} ", f" {new} ")
    # Shuffle sentences slightly
    sentences = text.replace("\n", " ").split(". ")
    random.shuffle(sentences)
    return ". ".join(sentences)


def create_document(name: str, text: str) -> str:
    """Write a document to disk and return filename."""
    filename = f"{name}.txt"
    filepath = OUTPUT_DOCS_DIR / filename
    filepath.write_text(text.strip(), encoding="utf-8")
    return filename


def main():
    OUTPUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    topic_keys = list(BASE_TEXTS.keys())
    pairs = []

    for i, topic in enumerate(topic_keys):
        base = BASE_TEXTS[topic]
        doc_id = str(i + 1).zfill(3)

        # Original document
        orig_name = create_document(f"doc_{doc_id}_original", base)

        # Copy (near-identical, label=1)
        copy_name = create_document(f"doc_{doc_id}_copy", base)
        pairs.append((orig_name, copy_name, 1))

        # Lightly mutated (high similarity, label=1)
        light_mut = mutate_word_order(base, swap_prob=0.05)
        light_mut_name = create_document(f"doc_{doc_id}_light_mutated", light_mut)
        pairs.append((orig_name, light_mut_name, 1))

        # Paraphrased (moderate similarity, label=1)
        para = paraphrase_text(base)
        para_name = create_document(f"doc_{doc_id}_paraphrased", para)
        pairs.append((orig_name, para_name, 1))

        # Heavy mutation (borderline, label=0)
        heavy_mut = mutate_word_order(paraphrase_text(base), swap_prob=0.3)
        heavy_mut_name = create_document(f"doc_{doc_id}_heavy_mutated", heavy_mut)
        pairs.append((orig_name, heavy_mut_name, 0))

    # Cross-topic (unrelated) negatives
    for i, topic_a in enumerate(topic_keys):
        for j, topic_b in enumerate(topic_keys):
            if j <= i:
                continue
            name_a = create_document(f"cross_{i}_{j}_a", BASE_TEXTS[topic_a])
            name_b = create_document(f"cross_{i}_{j}_b", BASE_TEXTS[topic_b])
            pairs.append((name_a, name_b, 0))

    # Shuffle
    random.shuffle(pairs)

    # Write CSV
    with open(OUTPUT_PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_a", "doc_b", "label"])
        for row in pairs:
            writer.writerow(row)

    pos = sum(1 for _, _, l in pairs if l == 1)
    neg = sum(1 for _, _, l in pairs if l == 0)

    print(f"Generated {len(pairs)} synthetic pairs:")
    print(f"  Positive (similar): {pos}")
    print(f"  Negative (dissimilar): {neg}")
    print(f"  Documents in: {OUTPUT_DOCS_DIR}/")
    print(f"  Pairs CSV: {OUTPUT_PAIRS_CSV}")
    print(f"\nNext step: python training/train_model.py")


if __name__ == "__main__":
    main()
