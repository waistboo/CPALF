# CPALF

Certified Pseudo-label Enhanced Active Learning Framework for Pattern Interest Evaluation (CPALF)

## Overview

Frequent Pattern Mining (FPM) on large graphs is a key task in graph data mining that aims to discover patterns whose support exceeds predefined thresholds. Traditional FPM methods often rely on single-dimensional evaluation metrics and do not account for subjective user preferences. To address this, CPALF is proposed to predict users' subjective preferences on patterns with minimal human interaction.

CPALF integrates:
- Active learning to efficiently collect user preference labels via human-machine interaction.
- Semi-supervised learning to generate high-confidence pseudo-labeled samples from unlabeled data.
- A replay strategy to mitigate catastrophic forgetting in incremental training.

Experimental results show CPALF can capture users' preferences effectively, achieving up to 96% prediction accuracy with limited labeled data.

## Dataset (Data sharing)

Our datasets are publicly available, the datasets are listed below:

- Twitter —
McAuley, J., & Leskovec, J. (2012). Learning to discover social circles in ego networks.
In Proceedings of the 25th international conference on neural information processing
systems (pp. 539–547).
- Twitch —
Rozemberczki, B., & Sarkar, R. (2021). Twitch Gamers: a dataset for evaluat-
ing proximity preserving and structural role-based node embeddings. arXiv,
2101.03091. 
- Skitter —
Leskovec, J., Kleinberg, J. M., & Faloutsos, C. (2005). Graphs over time: densification
laws, shrinking diameters and possible explanations. In Knowledge discovery and
data mining (pp. 177–187).
- Mico —
Elseidy, M., Abdelhamid, E., Skiadopoulos, S., & Kalnis, P. (2014). Grami: Frequent
subgraph and pattern mining in a single large graph. Proceedings of the VLDB
Endowment, 7(7), 517–528.
- Dblp —
ang, J., Leskovec, J.: Defining and evaluating network communities based on ground-
truth. In: Proceedings of the ACM SIGKDD Workshop on Mining Data Semantics. pp. 1–8
(2012)
- Aviation —
Elseidy, M., Abdelhamid, E., Skiadopoulos, S., Kalnis, P.: Grami: Frequent subgraph and
pattern mining in a single large graph. Proceedings of the VLDB Endowment 7(7), 517–
528 (2014)

Because the original data are large, the dataset files are provided via external download links.
Download link (xxx):


> Note: The datasets provided are public datasets; the uploaded archive contains our version of the dataset used in experiments.

## Repository Structure

- `dfs.py` — utilities to parse graph files in vertex/edge format and generate DFS codes per graph. Produces fixed-length encoded vectors for downstream processing.
- `Nearest_Neighbor.py` — build a nearest-neighbor graph from feature vectors, compute representativeness scores, and export PyTorch Geometric (`torch_geometric.data.Data`) graph objects.
- `try.py` — main CPALF implementation:
  - Pattern selection, clustering and representative sampling.
  - Pseudo-labeling pipeline (GCN-based propagation).
  - Semi-supervised training, distillation, and EWC regularization modules.
  - Active learning loop that selects instances based on representativeness and uncertainty.
## Key Components

1. DFS encoding (`dfs.py`)
   - Parse input files with lines starting with `v` (vertex) and `e` (edge).
   - Generate a deterministic DFS code for each pattern.
   - Map labels using an external encoding file and produce fixed-length vector outputs (CSV).

2. Nearest neighbor graph and representativeness (`Nearest_Neighbor.py`)
   - Compute k-nearest neighbors and create weighted edges.
   - Calculate per-node representativeness used for representative sampling in active learning.
   - Construct a nearest neighbor graph.
   - Save PyG data objects for GCN training.

3. Active learning & CPALF pipeline (`try.py`)
   - `PatternClassificationNN` backbones for feature-based models.
   - `GCN` module for label propagation and pseudo-label generation.
   - Semi-supervised training: supervised CE plus KL divergence for pseudo-labeled data.
   - `interactive_training_with_mixed_selection`: the active learning loop — combine representativeness and uncertainty for query selection, integrate pseudo-label filtering, and incremental training.

## Usage (example)

1. Prepare pattern vectors (CSV) using `dfs.py`:
   ```bash
   python dfs.py
   ```
   - Edit paths in `dfs.py` main block to point to your input files.

2. Build nearest-neighbor graph and save data objects:
   ```bash
   python Nearest_Neighbor.py
   ```

3. Run CPALF experiments:
   ```bash
   python GNNPLGtry.py
   ```
   - Modify dataset and config variables in `main()` to match local data paths and experiment settings.

## Notes on Reproducibility

- All hyperparameters (e.g., batch sizes, seeds, thresholds, number of iterations) are configurable in the `main()` sections of the scripts.
- When comparing acquisition or replay strategies, keep the rest of the pipeline identical for fair evaluation (model architecture, initialization, training hyperparameters, candidate pool size, query batch size, retraining schedule, and random seeds).
- If using GPU, ensure PyTorch and PyG are installed with CUDA support.

## Requirements

- Python 3.8+
- PyTorch
- torch-geometric
- scikit-learn
- numpy, scipy, networkx, matplotlib, pandas

Install common packages:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn numpy scipy networkx matplotlib pandas
```

## Contact & Citation

If you use this code, please cite the CPALF work (when published) and include the project README in your experiments. For questions, open an issue on the project repository.