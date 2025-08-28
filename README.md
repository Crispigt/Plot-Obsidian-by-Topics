# Obsidian Vault Topic Visualizer

Inspired by this: https://lmcinnes.github.io/datamapplot_examples/arXiv_math/

This script analyzes an Obsidian vault, identifies topical clusters among the notes, and generates an interactive 2D map for exploration. It uses the Qwen suite of models for high-quality embeddings and cluster summarization.

<img width="2801" height="1282" alt="Screenshot 2025-08-27 213551" src="https://github.com/user-attachments/assets/156d55c7-1caa-4d22-9df8-a6de84bf546f" />

<img width="2387" height="1253" alt="Screenshot 2025-08-27 213703" src="https://github.com/user-attachments/assets/fae3acc4-fb17-41be-8950-d5e34a30ed77" />

Should be noted that I am not done with this project and it's a work in progress though I've started school again. Please feel free to fork it and improve it :)

---


## Pipeline Overview
The entire process is automated in a single script:

Firstly the script recursively scans a specified directory for Markdown (.md) files. It then reads each file and strips away Markdown syntax (frontmatter, code blocks, links, etc.) to extract the raw text content.

Then each cleaned note is passed to a Qwen embedding model (Qwen/Qwen3-Embedding-4B). The model converts the text into a high-dimensional numerical vector (embedding) that captures its semantic meaning. Long notes are automatically chunked. 

After generating embeddings, the script uses BERTopic to perform semantic clustering and topic modeling.

BERTopic works by,
firstly applying UMAP dimensionality reduction (to 5D by default) to the embeddings,
then running HDBSCAN clustering on this reduced space to identify natural groupings,
on this it's using c-TF-IDF (class-based Term Frequency-Inverse Document Frequency) to analyze each cluster, identifying the words that are most distinctive to that topic compared to all other topics,
and it's also used for selecting the most representative documents from each cluster (those closest to the cluster centroid),
then for each identified cluster, the script then sends these distinctive keywords and representative documents to the Qwen LLM. This gives the LLM proper semantic context about what the cluster represents, allowing it to generate meaningful topic titles like "Quantum Mechanics" rather than just summarizing random document snippets.

Before BERTopic is used we also reduce down the embeddings to 2d through UMAP to get a map that we can plot everything on.

After all of this the script generates two interactive HTML files:

DataMapPlot Map: A clean, interactive plot where each point is a note. Points are colored by topic, and their size represents the note's file size or character count. You can hover to see note titles, search for notes, and click a point to open it directly in Obsidian.

Plotly Hull Map: A supplementary visualization that draws convex hulls around each cluster, making the topic boundaries more explicit.

---

## Features
Semantic Clustering: Automatically groups notes based on their content, not just tags or folders.

LLM-Powered Labeling: Uses a powerful LLM to generate human-readable names for each topic cluster.

Interactive 2D Map: Provides an intuitive way to explore the relationships between your notes.

Note Size Encoding: Point sizes visually represent the length or file size of your notes.

---

## Requirements
You can install the necessary Python packages using pip:

```Bash

pip install torch transformers bertopic umap-learn hdbscan pandas datamapplot plotly scikit-learn tqdm
```

Note: A CUDA-enabled GPU is highly recommended for performance, I used a RTX 3080 and for a vault with 1000+ notes it took 10+ min, for my own vault of 147 notes it took around 5 min.

---

## Usage
Run the script from your terminal, pointing it to your Obsidian vault.

```Bash

python obsidian_bertopic_datamapplot.py \
  --doc-root /path/to/your/ObsidianVault \
  --out-html YourVaultMap.html \
  --neighbors 24 \
  --min-dist 0.12 \
  --size-metric bytes
```

### Key Arguments:
--doc-root: (Required) The file path to your Obsidian vault's root directory.

--out-html: The name of the output HTML map file.

--embed-model: The embedding model to use from the Hugging Face Hub.

--label-model: The instruction-tuned LLM to use for summarizing cluster titles.

--neighbors / --min-dist: UMAP parameters to control the layout of the map. Adjust these to change cluster tightness.

--size-metric: The metric used to size the points on the map. Can be bytes (file size) or chars (text length).

The script will produce several output files, including the interactive HTML plots (.html), the embeddings (.npz), and the 2D coordinates (.csv).

Note: The pipeline works best with vaults containing 100+ notes. For larger collections, adjusting the --neighbors parameter (default: 24) can optimize the balance between local detail and global structure.


---

The visual at the top was generated with:

```Bash
python MapMyObsidian.py     --doc-root /mnt/c/Users/Documents/remote-vault     --neighbors 12 --min-dist 0.00     --embed-model Qwen/Qwen3-Embedding-4B     --label-model Qwen/Qwen3-4B-Instruct-2507 --min-samples 1 --min-cluster-size 5
```

---

Also another note on things I tried but didn't work for me, I tried using toponymy, but it didn't find any clusters for me, uncertain why, probably a error on my side. I also tried just creating the clusters with UMAP and HDBSCAN and then taking some examplar texts and creating the topics but they usually ended up to specific for the cluster.


