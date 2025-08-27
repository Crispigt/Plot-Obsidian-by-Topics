# Plot-Obsidian-by-Topics
A script that takes your obsidian notes and plots them based on semantical proximity. This is done localy through the qwen3-Embbedding-4B and then clustering is done with UMAP to get 2d vectors and then HDBSCAN for clusters. Then this is passed to BertTopic which uses qwen3-4B-Instruct to generate a Topic name for the cluster.
