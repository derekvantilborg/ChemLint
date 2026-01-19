# 🧬 MolML MCP Server

> **Molecular Machine Learning for AI Assistants** — An MCP server that gives LLMs native access to cheminformatics and molecular ML workflows

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-363%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MolML MCP** transforms AI assistants into powerful molecular machine learning workbenches. Through the Model Context Protocol (MCP), LLMs gain the ability to manipulate molecular structures, calculate descriptors, train ML models, and generate comprehensive analysis reports — all through natural conversation.

---

## ✨ Key Features

### 🧪 Molecular Operations
- **SMILES Processing**: Standardization, canonicalization, validation, cleaning pipelines
- **Molecular Descriptors**: Simple (MW, LogP, TPSA) and complex (ECFP, MACCS, RDKit fingerprints)
- **Scaffold Analysis**: Bemis-Murcko, generic scaffolds, cyclic skeletons
- **Similarity & Clustering**: Tanimoto similarity, DBSCAN, hierarchical, k-means, Butina clustering
- **Substructure Matching**: SMARTS pattern detection with 88+ built-in functional groups

### 🤖 Machine Learning
- **33 ML Algorithms**: Classification & regression (RF, GBM, SVM, linear models, ensembles with uncertainty)
- **Cross-Validation**: 6 strategies (k-fold, stratified, Monte Carlo, scaffold, cluster, leave-P-out)
- **Hyperparameter Tuning**: Grid search, random search
- **Model Evaluation**: 20+ metrics, confusion matrices, ROC curves, calibration plots

### 📊 Quality Reports
- **Data Quality Analysis**: 19-section comprehensive report (PAINS, Lipinski, duplicates, stereochemistry, etc.)
- **Split Quality Analysis**: 8 data leakage checks (duplicates, similarity, scaffolds, stereoisomers)
- **Scaffold Reports**: Diversity metrics, enrichment analysis, structural outliers

### 🔬 Advanced Features
- **Activity Cliff Detection**: Find structurally similar molecules with large activity differences
- **Dimensionality Reduction**: PCA, t-SNE for chemical space visualization
- **Statistical Analysis**: 15+ tests (t-test, ANOVA, correlation, normality tests)
- **Data Splitting**: Random, stratified, scaffold-based, cluster-based, temporal splits

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (included with uv)
- **uv** - Fast Python package installer ([install instructions](https://docs.astral.sh/uv/))
  ```bash
  # macOS/Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Or with pip
  pip install uv
  ```
- **Cairo** (optional) - Only needed for plotting features
  ```bash
  # macOS
  brew install cairo
  
  # Linux
  sudo apt-get install libcairo2-dev  # Ubuntu/Debian
  sudo yum install cairo-devel        # RHEL/CentOS
  ```

### Installation

**Option 1: Automated Installation (Recommended)**

Simply run the installer script - it will handle everything:

```bash
# Clone repository
git clone https://github.com/derekvantilborg/molml_mcp.git
cd molml_mcp

# Run installer
./install.sh
```

The installer will:
- ✅ Detect and verify uv installation
- ✅ Install Python dependencies
- ✅ Run server tests
- ✅ Detect Cairo (optional, for plotting - will warn if missing)
- ✅ Configure Claude Desktop automatically
- ✅ Provide clear next steps

**After installation, restart Claude Desktop:**
```bash
# macOS
pkill -x Claude && sleep 1 && open -a Claude

# Linux/Windows
# Restart Claude Desktop from your applications menu
```

**Option 2: Manual Configuration**

If you prefer manual setup or use a different MCP client:

```bash
# 1. Clone and install dependencies
git clone https://github.com/derekvantilborg/molml_mcp.git
cd molml_mcp
uv sync

# 2. Run tests to verify installation
uv run pytest -m server -q

# 3. Configure your MCP client
./mcp_client_configs/configure_claude.sh $(which uv) $(pwd) [cairo_path]
```

**Option 3: Other MCP Clients**

For clients other than Claude Desktop, add this to your MCP client configuration:

```json
{
  "mcpServers": {
    "molml-mcp": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "--directory",
        "/path/to/molml_mcp",
        "mcp",
        "run",
        "./src/molml_mcp/server.py"
      ],
      "enabled": true
    }
  }
}
```

### Usage with MCP Clients

Once installed, simply chat with your AI assistant naturally (Claude Desktop, Cline, etc.):

**Example Conversations:**

```
You: "Load my molecular dataset and check the quality"
AI: [Uses data_quality_analysis tool to generate comprehensive report]

You: "Clean the SMILES and remove salts"
AI: [Uses SMILES standardization pipeline]

You: "Calculate ECFP fingerprints and train a random forest classifier with 5-fold CV"
AI: [Calculates features, trains models, evaluates performance]

You: "Is there data leakage between my train and test splits?"
AI: [Uses data_split_quality_analysis to check for 8 types of leakage]

You: "What are the most common scaffolds in my dataset?"
AI: [Generates scaffold analysis report with diversity metrics]
```

---

## 📦 Architecture

### Core Components

```
molml_mcp/
├── tools/                      # 150+ molecular ML tools
│   ├── cleaning/              # SMILES cleaning, deduplication, label processing
│   ├── core/                  # Dataset ops, filtering, outliers, statistics
│   ├── core_mol/              # Scaffolds, similarity, activity cliffs, complexity
│   ├── featurization/         # Descriptors (simple, complex, SMILES encoding)
│   ├── ml/                    # Training, evaluation, CV, hyperparameter tuning
│   └── reports/               # Quality, scaffold, and split analysis reports
├── infrastructure/
│   ├── resources.py           # Manifest-based resource tracking
│   └── supported_resource_types.py  # CSV, model, JSON, PNG handlers
└── server.py                  # FastMCP server registration
```

### Resource Management

All data operations use a **manifest-based tracking system**:

- **Unique IDs**: Files named `{filename}_{8_HEX_ID}.{ext}` (e.g., `cleaned_data_A3F2B1D4.csv`)
- **Manifest JSON**: Tracks all resources with metadata (created_at, created_by, explanation)
- **Type Registry**: Handlers for CSV (pandas), models (joblib), JSON, PNG (matplotlib)
- **Project Isolation**: Each project has its own manifest and resource directory

---

## 🛠️ Tool Categories

### 1️⃣ Data Cleaning & Preparation

**Production-Ready Molecular Data Cleaning** — The most comprehensive SMILES standardization pipeline available, handling edge cases that break most cheminformatics workflows.

#### 🔧 Core Cleaning Pipeline

**`default_SMILES_standardization_pipeline`** — Battle-tested 10-step cleaning process:

1. **SMILES Validation** → Catches malformed structures before they poison your dataset
2. **Salt/Counterion Removal** → Strips 50+ common salts (NaCl, HCl, TFA, etc.) and counterions
3. **Solvent Removal** → Removes 17 common solvents (water, DMSO, ethanol, etc.)
4. **Largest Fragment Selection** → Keeps parent compound, discards fragments
5. **Metal Disconnection** → Safely handles organometallics, breaks problematic metal bonds
6. **Charge Neutralization** → Converts charged species to neutral forms (preserves zwitterions)
7. **Stereochemistry Handling** → Flatten to remove stereo OR keep specified stereochemistry
8. **Isotope Normalization** → Remove isotope labels (D, ¹³C, etc.) or preserve them
9. **Tautomer Canonicalization** → RDKit MolStandardize for consistent tautomeric forms
10. **Final Canonicalization** → Consistent SMILES representation for deduplication

**Output**: Clean dataset + detailed comment columns documenting every transformation

#### 🔍 Advanced Cleaning Tools

- **`find_duplicates_dataset`**: 
  - Detects exact SMILES duplicates (after canonicalization)
  - **Activity conflict detection**: Flags molecules with inconsistent bioactivity values
  - Reports fold-differences for regression, class mismatches for classification
  - Aggregation strategies: mean, median, majority vote, keep all, keep first

- **Drug-Likeness Filters**:
  - `filter_by_pains`: 480 Pan-Assay Interference patterns (nuisance compounds)
  - `filter_by_lipinski_ro5`: Rule of Five (MW, LogP, H-bonds) for oral bioavailability
  - `filter_by_veber_rules`: TPSA + rotatable bonds for absorption prediction
  - `filter_by_lead_likeness`: Optimized ranges for lead compounds
  - `filter_by_rule_of_three`: Fragment-like compounds for screening
  - `filter_by_qed`: Quantitative Estimate of Drug-likeness (0-1 score)

- **Property-Based Filtering**:
  - `filter_by_property_range`: Custom thresholds for any molecular property
  - `filter_by_scaffold`: Keep/remove molecules matching specific scaffolds
  - `filter_by_functional_groups`: Include/exclude based on 88+ functional groups

#### 📊 Quality Assessment Before/After

**`data_quality_analysis`** generates 19-section reports covering:
- ✅ SMILES validity rate (parseable structures)
- ✅ Salt/fragment/solvent contamination levels
- ✅ Stereochemistry completeness (specified vs unspecified)
- ✅ Charge state distribution (neutral, cations, anions, zwitterions)
- ✅ Organometallic compounds and non-standard isotopes
- ✅ PAINS pattern prevalence
- ✅ Duplicate rate and activity conflicts
- ✅ Drug-likeness compliance (Lipinski, Veber, QED)
- ✅ Physicochemical property distributions
- ✅ Scaffold diversity metrics (Gini, Shannon entropy)
- ✅ **Automated cleaning recommendations** ranked by priority

**Real-World Impact**: 
- Handles messy vendor data (Enamine, Mcule, ZINC) without manual intervention
- Processes ChEMBL/PubChem exports with complex stereochemistry
- Rescues "dirty" legacy datasets that crashed RDKit workflows
- Reduces false positives in QSAR models by removing assay interference compounds

### 2️⃣ Molecular Descriptors
- **Simple**: MW, LogP, TPSA, H-bonds, rotatable bonds, QED (10+ descriptors)
- **Complex**: ECFP, FCFP, MACCS, RDKit fingerprints, atom pair, topological
- **Encoding**: Integer encoding, one-hot encoding, learned embeddings

### 3️⃣ Machine Learning
- **Training**: Single model or cross-validation with 6 strategies
- **Algorithms**: 33 models (11 classifiers, 11 regressors, 6 w/ uncertainty each)
- **Evaluation**: Classification metrics, regression metrics, ensemble evaluation
- **Hyperparameter Tuning**: 3 search strategies with customizable spaces

### 4️⃣ Analysis & Reports
- **Data Quality**: 19-section report (completeness, validity, drug-likeness, diversity)
- **Split Quality**: 8 leakage checks (exact, similarity, scaffold, stereoisomer)
- **Scaffold Analysis**: Distribution, diversity (Gini, Shannon), enrichment, outliers

### 5️⃣ Cheminformatics
- **Scaffolds**: Bemis-Murcko, generic, cyclic skeletons with diversity metrics
- **Similarity**: Tanimoto, Dice, pairwise matrices, nearest neighbors
- **Activity Cliffs**: Regression/classification cliff detection with statistics
- **Complexity**: Bertz, fragment, shape, BalabanJ indices

---

## 📚 Example Workflows

### Workflow 1: Data Quality Assessment → Cleaning → ML

```python
# 1. Check data quality
"Generate a quality report for my dataset"
→ data_quality_analysis() creates 19-section report

# 2. Clean based on recommendations
"Clean SMILES, remove PAINS, and filter by Lipinski rules"
→ default_SMILES_standardization_pipeline()
→ filter_by_pains()
→ filter_by_lipinski_ro5()

# 3. Train models
"Calculate ECFP4 fingerprints and train random forest with scaffold-based CV"
→ calculate_ecfp_fingerprints()
→ train_ml_models_cross_validation(cv_strategy='scaffold')
```

### Workflow 2: Split Quality Check → Activity Cliff Analysis

```python
# 1. Check for data leakage
"Is there leakage between my train and test splits?"
→ data_split_quality_analysis() checks 8 leakage types

# 2. Find activity cliffs
"Find activity cliffs in my training set"
→ detect_activity_cliffs() identifies similar molecules with large activity differences
```

### Workflow 3: Scaffold Diversity → Clustering → Visualization

```python
# 1. Analyze scaffolds
"What are the most common scaffolds and how diverse is my dataset?"
→ scaffold_analysis() with diversity metrics

# 2. Cluster molecules
"Cluster molecules by Tanimoto similarity using DBSCAN"
→ cluster_dbscan_on_similarity()

# 3. Visualize chemical space
"Reduce dimensions with t-SNE for visualization"
→ reduce_dimensions_tsne()
```

---

## 🧪 Testing

```bash
# Run all tests (363 tests)
python -m pytest -v

# Run specific test modules
pytest tests/tools/ml/test_training.py -v
pytest tests/tools/reports/test_quality.py -v

# Run with coverage
pytest --cov=molml_mcp --cov-report=html
```

**Test Coverage:**
- ✅ 363 tests across all modules
- ✅ Infrastructure (resource management, manifests)
- ✅ Cleaning & filtering tools
- ✅ Molecular operations (scaffolds, similarity, activity cliffs)
- ✅ ML training, evaluation, and cross-validation
- ✅ Report generation (quality, scaffold, split analysis)

---

### Deployment

```bash
# Quick deploy (updates MCP client config automatically)
./deploy_mcp_server.sh

# Manual deployment
uv mcp install src/molml_mcp/server.py
# Restart your MCP client (Claude Desktop, Cline, etc.)
```

---

## 📖 Documentation

- **Tool Discovery**: All tools have enhanced docstrings with 🚀 markers for primary functions
- **Guides**: TODO

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

Built with:
- **FastMCP** - MCP server framework
- **RDKit** - Cheminformatics toolkit
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

---

## 📬 Contact

**Derek van Tilborg** - [@derekvantilborg](https://github.com/derekvantilborg)

**Project Link**: [https://github.com/derekvantilborg/molml_mcp](https://github.com/derekvantilborg/molml_mcp)

---

<div align="center">

**Give your AI assistant superpowers for molecular machine learning** 🚀

</div>
