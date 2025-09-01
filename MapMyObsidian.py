#!/usr/bin/env python3
"""
Obsidian → Qwen embeddings → BERTopic (UMAP + HDBSCAN)
→ QwenChatLLM cluster title summarization
→ DataMapPlot interactive map (light theme, crisp) + Plotly hull map,
with point sizes encoding file size or text length.

Example:
  python MapMyObsidian.py \
    --doc-root /path/to/ObsidianVault \
    --out-html obsidian_map_datamapplot.html \
    --neighbors 24 --min-dist 0.12 \
    --embed-model Qwen/Qwen3-Embedding-4B \
    --label-model Qwen/Qwen3-4B-Instruct-2507 \
    --size-metric bytes
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import inspect
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import umap
import datamapplot  # pip install datamapplot

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from bertopic import BERTopic
import hdbscan



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    # logging.info("Cuda available.")
    torch.cuda.manual_seed_all(SEED)

#Strip md files of random chars
FM   = re.compile(r"^---.*?---\s*", re.S)
CODE = re.compile(r"```.*?```", re.S)
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
FMT  = re.compile(r"[*_#>~-]{1,}")

def md_to_text(md: str) -> str:
    md = FM.sub("", md)
    md = CODE.sub(" ", md)
    md = LINK.sub(r"\1", md)
    md = FMT.sub(" ", md)
    md = re.sub(r"\s+", " ", md)
    #logging.info("Stripped files of md ")
    return md.strip()


#Wrapper for Qwen so that it interacts as a chatbot/actually answers requests
#And also wraps all of it's actions in one class
class QwenChatLLM:
    def __init__(self, model_name: str, system: str = "",
                 max_new_tokens: int = 32, temperature: float = 0.0,
                 batch_size: int = 16, load_in_4bit: bool = False):
        self.model_name = model_name
        self.supports_system_prompts = True
        self.supports_batched_generation = True

        self.system = (system or "").strip()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id

        kwargs = dict(trust_remote_code=True, device_map="auto")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def _format(self, user_prompt: str) -> str:
        if hasattr(self.tok, "apply_chat_template"):
            msgs = []
            if self.system:
                msgs.append({"role": "system", "content": self.system})
            msgs.append({"role": "user", "content": user_prompt})
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"{self.system}\n\n{user_prompt}" if self.system else user_prompt

    def _post(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = (s.replace("\u00AD", "")     # soft hyphen
             .replace("\u200B", "")      # zero width space
             .replace("\u200C", "")      # ZWNJ
             .replace("\u200D", ""))     # ZWJ
        s = re.sub(r"[\u2010-\u2015]", "-", s)

        s = re.sub(r"[\r\n\t]+", " ", s).strip().strip("“”\"'‘’[](){}:;,.")
        words = re.findall(r"[^\W_]+(?:-[^\W_]+)*", s, flags=re.UNICODE)
        text = " ".join(words[:5]).strip()
        return text or s[:80]

    def __call__(self, prompts, **gen_kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = [self._format(p) for p in prompts]

        outs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.model.device)
            in_lens = (enc.input_ids != self.tok.pad_token_id).sum(dim=1).tolist()

            do_sample = bool(gen_kwargs.get("do_sample"))
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=float(gen_kwargs.get("temperature", self.temperature)) if do_sample else None,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
            )
            for row, L in zip(gen, in_lens):
                text = self.tok.decode(row[L:], skip_special_tokens=True).strip()
                outs.append(self._post(text))
        return outs


# Qwen embedding wrapper
class QwenEmbedding:
    def __init__(
        self,
        model_name_or_path: str,
        use_cuda: bool = True,
        max_length: int = 1024,
        batch_size: int = 16,
    ) -> None:
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        if self.device == "cuda":
            use_bf16 = torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if use_bf16 else torch.float16
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        else:
            self.dtype = torch.float32
            attn_impl = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, padding_side="left"
        )

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
        )
        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, device_map="auto", **model_kwargs
            )
        except Exception:
            self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
            self.model = self.model.to(self.device)
        self.model.eval()

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.max_length = max_length
        self.batch_size = batch_size
        self.has_encode = hasattr(self.model, "encode") or hasattr(
            self.model, "get_text_embedding"
        )

    @staticmethod
    def _masked_mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_states)
        summed = (last_hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def encode(
        self,
        sentences: Union[List[str], str],
        is_query: bool = False,
        instruction: str = "Represent this sentence for searching relevant passages:",
    ) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, s) for s in sentences]

        outs: List[torch.Tensor] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                if self.has_encode:
                    if hasattr(self.model, "encode"):
                        pooled = self.model.encode(**inputs)
                    else:
                        pooled = self.model.get_text_embedding(**inputs)
                else:
                    h = self.model(**inputs).last_hidden_state
                    pooled = self._masked_mean_pool(h, inputs["attention_mask"])
                outs.append(F.normalize(pooled, p=2, dim=1))
        return torch.cat(outs, dim=0).to(torch.float32)

    def embed_long_markdown(self, text: str, chunk_tokens: int = 512, overlap: int = 64) -> torch.Tensor:
        tok = self.tokenizer(text, return_tensors="pt", truncation=False)
        ids = tok["input_ids"][0]
        if len(ids) <= chunk_tokens:
            return self.encode(text)[0]
        chunks: List[str] = []
        step = max(1, chunk_tokens - overlap)
        for start in range(0, len(ids), step):
            piece = ids[start : start + chunk_tokens]
            if len(piece) == 0:
                break
            chunk_text = self.tokenizer.decode(piece, skip_special_tokens=True)
            chunks.append(chunk_text)
        embs = self.encode(chunks)
        doc = F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=1)[0]
        return doc


GENERIC_NAMES = {"misc", "miscellaneous", "general", "other", "uncategorized", "unknown", "topic", "notes"}

def clean_label(label: str) -> str:
    label = str(label or "").strip().strip("\"'：:.-–—")
    label = re.sub(r"[\r\n]+", " ", label)
    label = re.sub(r"\s+", " ", label)
    return " ".join(label.split()[:6])

def scale_sizes(values: np.ndarray, min_size: float, max_size: float, log=True) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if log:
        v = np.log1p(v)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo + 1e-12:
        return np.full_like(v, (min_size + max_size) / 2.0)
    return (v - lo) / (hi - lo) * (max_size - min_size) + min_size


def list_markdown_files(
    root: Path,
    exclude_dirs: Iterable[str] = ("excalidraw", ".obsidian", ".git", "node_modules"),
) -> List[Path]:
    ex = {d.lower() for d in exclude_dirs}
    files: List[Path] = []
    for p in root.rglob("*.md"):
        parts = {part.lower() for part in p.parts}
        if any(x in ex for x in parts):
            continue
        files.append(p)
    return files




def main() -> None:
    parser = argparse.ArgumentParser(description="Obsidian + Qwen + BERTopic + DataMapPlot (summarized titles, sized points)")
    parser.add_argument("--doc-root", type=str, required=True)
    parser.add_argument("--out-html", type=str, default="obsidian_map_datamapplot.html")
    parser.add_argument("--out-embeddings", type=str, default="obsidian_qwen_embeddings.npz")
    parser.add_argument("--out-csv", type=str, default="obsidian_qwen_coords.csv")

    parser.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--label-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)

    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)

    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--min-dist", type=float, default=0.12)

    parser.add_argument("--min-cluster-size", type=int, default=6)
    parser.add_argument("--min-samples", type=int, default=2)

    parser.add_argument("--size-metric", type=str, default="bytes", choices=["bytes", "chars"],
                        help="Size encoding for points: file size in bytes or cleaned text length in chars")

    parser.add_argument("--max-file-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--cluster-dim", type=int, default=5,
                        help="UMAP dimensionality used for HDBSCAN/BERTopic (not the 2D map)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    root = Path(args.doc_root).expanduser().resolve()
    assert root.exists(), f"Doc root not found: {root}"

    out_html = args.out_html
    if args.out_html is None or out_html == "obsidian_map_datamapplot.html":
        out_html = root.name + "_plot"

    logging.info("Step 1: Scan markdown files")
    md_files = list_markdown_files(root)
    logging.info(f"Found {len(md_files)} markdown files")

    documents: List[str] = []
    filenames: List[str] = []
    doc_lengths_chars: List[int] = []
    file_sizes_bytes: List[int] = []

    for p in tqdm(md_files, desc="Loading docs"):
        try:
            st = p.stat()
            if st.st_size > args.max_file_bytes:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            clean = md_to_text(text)
            documents.append(clean)
            filenames.append(str(p.resolve()))
            doc_lengths_chars.append(len(clean))
            file_sizes_bytes.append(st.st_size)
        except Exception as e:
            logging.warning(f"Skipping {p}: {e}")

    if not documents:
        logging.error("No documents to process. Exiting.")
        return

    if args.size_metric == "bytes":
        size_raw = np.array(file_sizes_bytes, dtype=float)
    else:
        size_raw = np.array(doc_lengths_chars, dtype=float)

    logging.info("Step 2: Embed with Qwen")
    embedder = QwenEmbedding(args.embed_model, max_length=args.max_length, batch_size=args.batch_size)

    embs: List[torch.Tensor] = []
    for doc in tqdm(documents, desc="Embedding"):
        vec = embedder.embed_long_markdown(doc, chunk_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
        embs.append(vec.cpu())
    X = torch.stack(embs, dim=0).numpy()
    np.savez(args.out_embeddings, X=X, files=np.array(filenames, dtype=object))
    logging.info(f"Saved embeddings to {args.out_embeddings} with shape {X.shape}")

    from sklearn.decomposition import PCA

    X = np.ascontiguousarray(X.astype(np.float32))
    n_samples, n_features = X.shape

    pca_dim = max(2, min(256, n_samples - 1, n_features))

    svd_solver = "randomized" if (n_features > 1000 and n_samples < 500) else "auto"
    logging.info(f"PCA: n_samples={n_samples}, n_features={n_features}, n_components={pca_dim}, solver={svd_solver}")

    pca = PCA(n_components=pca_dim, random_state=SEED, svd_solver=svd_solver)
    X_pca = pca.fit_transform(X)

    reducer_2d = umap.UMAP(
        n_neighbors=max(6, min(40, args.neighbors)),
        min_dist=max(0.03, min(0.3, args.min_dist)),
        n_components=2,
        metric="cosine",
        random_state=SEED,
    )
    XY = reducer_2d.fit_transform(X_pca)

    df_coords = pd.DataFrame({"x": XY[:, 0], "y": XY[:, 1], "filename": filenames,
                              "doc_len_chars": doc_lengths_chars, "file_size_bytes": file_sizes_bytes})
    df_coords.to_csv(args.out_csv, index=False)
    logging.info(f"Saved coordinates CSV to {args.out_csv}")

    logging.info("Step 4: BERTopic clustering + Qwen summarization")

    umap_model_topics = umap.UMAP(
        n_neighbors=max(12, min(40, args.neighbors)),
        min_dist=max(0.03, min(0.3, args.min_dist)),
        n_components=max(2, args.cluster_dim),   # e.g., 5
        metric="cosine",
        random_state=SEED,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=None,
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model_topics,
        hdbscan_model=hdbscan_model,
        language="multilingual",
        calculate_probabilities=False,
        verbose=args.verbose,
    )

    topics, _ = topic_model.fit_transform(documents, embeddings=X)
    n_clusters_last = len(set([t for t in topics if t != -1]))

    logging.info(f"Clusters: {n_clusters_last} | Outliers: {(np.array(topics) == -1).sum()}")

    SUM_SYS = (
        "You write short, specific topic titles for clusters of notes. "
        "Return ONLY 2–4 words in Title Case. No punctuation or quotes. "
        "Avoid generic words like misc, general, other."
    )
    label_llm = QwenChatLLM(
        model_name=args.label_model,
        system=SUM_SYS,
        max_new_tokens=24,
        temperature=0.1,
        batch_size=args.batch_size,
    )

    topic_labels: Dict[int, str] = {}
    unique_topics = sorted(set(t for t in topics if t != -1))

    for t in unique_topics:
        words = [w for w, _ in (topic_model.get_topic(t) or [])][:6]

        reps = (topic_model.get_representative_docs(t) or [])[:6]
        snippets = []
        for d in reps:
            s = md_to_text(d)
            snippets.append(s[:400]) 
        examples = "\n".join(f"<NOTE>\n{snip}\n</NOTE>" for snip in snippets)

        prompt = (
            f"Keywords: {', '.join(words)}\n"
            f"Examples:\n{examples}\n\n"
            "Return the title now."
        )

        try:
            title = clean_label(label_llm(prompt)[0])
        except Exception:
            title = f"Topic {t}"

        if not title or title.lower() in GENERIC_NAMES:
            title = f"Topic {t}"

        topic_labels[t] = title

    topic_model.set_topic_labels(topic_labels)

    labels_for_docs = np.array(
        ["Outliers" if t == -1 else topic_labels.get(t, f"Topic {t}") for t in topics],
        dtype=object
    )
    display_label_layers = [labels_for_docs]

    logging.info(f"Unique labels going into DataMapPlot:{np.unique(labels_for_docs)}")

    from urllib.parse import quote
    stems = [Path(f).stem for f in filenames]
    hover_text = np.array(stems, dtype=object)
    obsidian_urls = np.array(["obsidian://open?path=" + quote(f) for f in filenames], dtype=object)

    extra_point_data = pd.DataFrame({
        "title": hover_text,
        "obsidian_url": obsidian_urls,
        "path": filenames,
        "size_raw": size_raw,
    })

    label_layers_for_plot = display_label_layers

    dmp_sizes = scale_sizes(size_raw, min_size=2.0, max_size=36.0)

    dmp_sig = inspect.signature(datamapplot.create_interactive_plot)
    dmp_kwargs = dict(
        hover_text=hover_text,
        enable_search=True,
        search_field="hover_text",
        extra_point_data=extra_point_data,
        on_click="window.open('{obsidian_url}', '_blank')",
        title="Obsidian Topic Map (Qwen + BERTopic)",
        sub_title=f"{len(filenames)} notes · {n_clusters_last} clusters",
        initial_zoom_fraction=0.95,
        noise_label="Outliers",
        marker_size_array=dmp_sizes.astype(float).tolist(),
        point_radius_min_pixels=2,
        point_radius_max_pixels=36,
        point_size_scale=1.0,
    )
    if "darkmode" in dmp_sig.parameters:
        dmp_kwargs["darkmode"] = False


    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    uniq = pd.Index(label_layers_for_plot[0]).unique().tolist()
    uniq = [u for u in uniq if u != "Outliers"] + ["Outliers"]

    cmap = cm.get_cmap("tab20", len(uniq))
    label_color_map = {lab: mcolors.to_hex(cmap(i)) for i, lab in enumerate(uniq)}
    label_color_map["Outliers"] = "#999999"

    dmp_kwargs["label_color_map"] = label_color_map
    dmp_kwargs.pop("cmap", None)
    dmp_kwargs.pop("palette_theta_range", None)

    try:
        plot = datamapplot.create_interactive_plot(
            XY, *label_layers_for_plot, **dmp_kwargs
        )
    except ValueError as e:
        if "array of sample points is empty" in str(e):
            import copy, logging as _log
            _log.warning("Palette interpolation failed; falling back to default palette.")
            safe_kwargs = copy.deepcopy(dmp_kwargs)
            safe_kwargs.pop("cmap", None)
            safe_kwargs.pop("palette_theta_range", None)
            plot = datamapplot.create_interactive_plot(
                XY, *label_layers_for_plot, **safe_kwargs
            )
        else:
            raise

    
    plot.save(out_html)
    logging.info(f"Saved interactive DataMapPlot to {out_html}")

    # Have this mostly just to be able to see where it actualy drew the clusters
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull

    def build_plot(df: pd.DataFrame, title="Obsidian Topic Map (Plotly + hulls)", outliers_label="Outliers"):
        fig = px.scatter(
            df,
            x="x", y="y",
            color="cluster",
            size="size_vis",
            hover_name="filename",
            labels={"x": "UMAP Dim 1", "y": "UMAP Dim 2"},
            template="plotly_white",
            render_mode="webgl",
            size_max=40,
            color_discrete_map={outliers_label: "lightgrey"},
        )
        for tr in fig.data:
            if tr.name == outliers_label or tr.name == f"cluster={outliers_label}":
                try:
                    tr.marker.opacity = 0.35
                    tr.marker.size = 6
                    tr.marker.color = "lightgrey"
                except Exception:
                    pass
        for topic in df["cluster"].unique():
            if topic == outliers_label:
                continue
            pts = df[df["cluster"] == topic][["x", "y"]].values
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
                hp = np.append(pts[hull.vertices], [pts[hull.vertices][0]], axis=0)
                fig.add_trace(go.Scatter(
                    x=hp[:, 0], y=hp[:, 1], mode="lines",
                    line=dict(width=2),
                    name=f"{topic} boundary", showlegend=False
                ))
            except Exception:
                pass
        fig.update_layout(legend_title_text="Cluster Topic", title_x=0.5)
        fig.update_traces(
            customdata=df["obsidian_url"],
            hovertemplate="<b>%{hovertext}</b><br>%{x:.3f}, %{y:.3f}<extra></extra>"
        )
        return fig

    from urllib.parse import quote as _quote
    obsidian_urls_list = ["obsidian://open?path=" + _quote(f) for f in filenames]
    size_vis_plotly = scale_sizes(size_raw, min_size=4.0, max_size=80.0)

    df_plotly = pd.DataFrame({
        "x": XY[:, 0], "y": XY[:, 1],
        "cluster": label_layers_for_plot[0],
        "filename": filenames,
        "size_vis": size_vis_plotly,
        "obsidian_url": obsidian_urls_list,
    })
    fig = build_plot(df_plotly)
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    try:
        div_id = fig.get_div_id()
        html = html.replace("</body>", f"""
<script>
(function(){{
  var gd = document.getElementById('{div_id}');
  if (!gd || !gd.on) return;
  gd.on('plotly_click', function(data){{
    try {{
      var url = data.points && data.points[0] && data.points[0].customdata;
      if (url) window.open(url, '_blank');
    }} catch(e){{ console.error(e); }}
  }});
}})();
</script></body>""")
    except Exception:
        pass
    plotly_path = Path(out_html).with_suffix(".plotly.html")
    plotly_path.write_text(html, encoding="utf-8")
    logging.info(f"Saved Plotly hull map to {plotly_path}")

    topic_names_per_layer = [
        sorted(set([str(x) for x in display_label_layers[0] if str(x).strip() and str(x) != "Unlabelled"]))
    ]
    Path(out_html).with_suffix(".layers.json").write_text(
        json.dumps(topic_names_per_layer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Saved topic name layers JSON")


if __name__ == "__main__":
    main()
