# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model. Created for the UW Startups S23 Kickoff event with guidance from [Jacky Zhao](https://jzhao.xyz/) and [Sean Ashley](https://www.linkedin.com/in/sean-ashley). [Simha Kalimipalli](https://github.com/Simha-Kalimipalli) later aded interactivity!

![Sample output of script](https://github.com/ansonyuu/matchmaking/blob/main/sample.png?raw=true)

## Instructions for use

1. Collect or format your data in the following format

| Name  | What are your interests? (or varying permutations of this question) |
| ----- | ------------------------------------------------------------------- |
| Alice | I love being the universal placeholder for every CS joke ever       |
| Bob   | I too love being the universal placeholder for every CS joke        |

2. Clone the repository
3. Install all required packages using pip or conda:

- `umap-learn`
- `scikit-learn`
- `scipy`
- `sentence-transformers`
- `matplotlib`
- `pyvis`
- `pandas`
- `numpy`
- `seaborn`
- `branca`

4. Replace `attendees.csv` in `visualizer.ipynb` with the path to your downloaded data
5. Run all cells
6. Bask in the glory of having an awesome new poster
7. Make two (!) cool interactive visualizations


## What Are Embeddings?

When we read a sentence like “I enjoy hiking and swimming,” we immediately understand what it means. A computer, however, does not naturally understand language. To a computer, words are just symbols. Embeddings are a way of translating words and sentences into numbers so that a computer can compare and understand their meaning.

Instead of storing a word as just text, an embedding represents it as a position in an abstract space — like placing ideas on a map. On this “meaning map,” words or sentences that express similar ideas are placed closer together. The actual numbers are not important to humans, but the distances between them are.

For example, in our classmates dataset, one person might write “I like hiking, swimming, and enjoying nice weather.” Another might say “I enjoy being outdoors in nature and trail running.” Although the wording is different, both describe outdoor activities. When converted into sentence embeddings, these two descriptions end up close together on the meaning map. The system then recognizes that these classmates likely share similar interests.

On the other hand, someone who writes “I love basketball and video games” would appear farther away from the outdoor group, because their interests are different.

Sentence embeddings work by looking at patterns learned from large amounts of text. The model has seen how words are used together in many contexts, so it learns that “hiking,” “nature,” and “outdoors” are related concepts. When it processes a new sentence, it captures this relationship numerically.

Once every classmate’s description is converted into this numerical form, we can measure how close they are to each other. If two descriptions are close in this space, the system assumes they have similar meaning. This is how the matchmaking model suggests top matches.

In simple terms, embeddings allow a computer to compare meaning instead of just matching identical words. They turn language into something measurable, while still preserving the relationships between ideas.

## Data Analysis

To evaluate how sensitive sentence embeddings are to changes in the dataset, I modified three sentences in classmates.csv and compared the original and updated embeddings using cosine similarity.

For Greg Kirczenow, I made a minor change by replacing “bike” with the synonym “cycle” in the sentence “Swim, bike, run.” The cosine similarity between the original and modified embedding was 0.843. Although this change did not alter the overall meaning, it still produced a noticeable shift in the embedding space. This suggests that even small wording changes can influence the numerical representation, especially when the sentence is short and each word carries significant weight.

For Mohammad Pakdoust, I introduced a major semantic change by significantly altering the activities described, shifting the focus to a different topic. The cosine similarity dropped to 0.244, indicating a large movement in the embedding space. This confirms that embeddings are highly sensitive to changes in meaning rather than just surface wording.

For Soundarya Venkataraman, I made another major change by reversing the sentiment toward outdoor activities (e.g., from enjoying the outdoors to avoiding it). The similarity score was 0.328, again reflecting a substantial shift.

Overall, these results demonstrate that sentence embeddings are relatively robust to minor lexical substitutions but respond strongly when the core meaning changes. This highlights how small dataset idiosyncrasies can affect downstream similarity results and ultimately influence matchmaking outcomes.


