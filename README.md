#Matchmaking with Sentence Embeddings#

This project explores how sentence embeddings can be used to measure similarity between classmates based on short descriptions of their interests. The goal is to analyze how data changes and model choice impact similarity rankings in a simple matchmaking pipeline.

##The pipeline consists of three components:##

- Data – Classmate names and interest descriptions (classmates.csv)

- Embedding Model – Converts sentences into numerical representations

- Similarity & Ranking – Computes cosine similarity and ranks classmates

- Setup

- ### Create and activate environment (example using uv):

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt


- ### Run the baseline embedding pipeline:

uv run python main.py


This generates:

embeddings.json

What Are Embeddings?

Sentence embeddings are a way to represent the meaning of text as numbers so that a computer can compare them. Instead of storing words as plain text, embedding models convert sentences into lists of numbers that capture semantic meaning.

For example:

“Swim, bike, run”

“Swim, cycle, run”

Even though the wording differs slightly (“bike” vs “cycle”), the meanings are very similar. An embedding model places these sentences close together in a conceptual meaning space.

However, if we change the sentence to:

“I avoid outdoor activities and prefer staying inside.”

The meaning shifts significantly. As a result, the embedding will move far away in semantic space.

Embeddings allow us to compute similarity between classmates using cosine similarity. If two embeddings are close, their interests are considered similar. This allows us to build a simple matchmaking system based on meaning rather than exact word matches.

# Data Analysis

To evaluate how sensitive embeddings are to dataset changes, I modified three sentences in classmates.csv:

Minor change: Replaced “bike” with “cycle” in “Swim, bike, run.”

Major change: Significantly altered the activities described for one classmate.

Major sentiment change: Reversed the preference toward outdoor activities for another classmate.

After regenerating embeddings and comparing them to the original embeddings using cosine similarity:

Greg Kirczenow → 0.843 (moderate shift)

Mohammad Pakdoust → 0.244 (large shift)

Soundarya Venkataraman → 0.328 (large shift)

These results show that embeddings are relatively robust to small wording changes but highly sensitive to changes in meaning. Even subtle dataset variations can influence similarity scores and matchmaking outcomes.

Embedding Sensitivity Tests

To evaluate the impact of model choice, I compared:

Baseline: all-MiniLM-L6-v2

Alternative: all-mpnet-base-v2

Using both models, I ranked classmates by cosine similarity relative to the anchor person (Mohammad Pakdoust) and computed Spearman’s rank correlation between the two ranking lists.

Spearman ρ = 0.3971 (p = 0.1145)

This indicates relatively low agreement between rankings.

Significant rank changes included:

Jeevan Dhakal → −10 positions

Somto Muotoe → +10 positions

Nikola Kriznar → +8 positions

The top 5 matches differed noticeably between the two models.

These findings demonstrate that embedding model choice can materially affect downstream similarity-based matchmaking results. Different models capture semantic relationships differently, leading to meaningful changes in ranking behavior even when the dataset remains unchanged.

Reproducibility

Baseline model:

sentence-transformers/all-MiniLM-L6-v2


Alternative model:

sentence-transformers/all-mpnet-base-v2


To compare models:

uv run python model_comparison.py --anchor "Mohammad Pakdoust"
