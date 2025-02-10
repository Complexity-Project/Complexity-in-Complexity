# Complexity in Complexity: Understanding Visual Complexity Through Structure, Color, and Surprise

[![arXiv](https://img.shields.io/badge/arXiv-2501.15890-B31B1B.svg)](https://arxiv.org/abs/2501.15890)

This repository contains code, data, and scripts for the paper:

> **Complexity in Complexity: Understanding Visual Complexity Through Structure, Color, and Surprise**  
> Karahan Sarıtaş, Peter Dayan, Tingke Shen, Surabhi S. Nath  
> *arXiv preprint arXiv:2501.15890, 2025*

---

## Overview

We present novel interpretable features for **visual complexity**, combining:

1. **Multi-Scale Sobel Gradient (MSG)** to capture **structural intensity** variations in images,  
2. **Multi-Scale Unique Color (MUC)** to quantify **colorfulness** at multiple scales,  
3. **Surprise Scores** derived from **Large Language Models**, indicating **unusual/surprising** objects or contexts.

Using these features alongside existing segmentation/object-based features, we demonstrate improved performance in predicting **human-rated visual complexity** across multiple datasets. 



---

## Quick Start

So, do you want to reproduce our results? Here's how you can do it in a few simple steps:
1. **Clone and Install**

```bash
git clone https://github.com/Complexity-Project/Complexity-in-Complexity.git
cd Complexity-in-Complexity
pip install -r requirements.txt
```

2. **Run Experiments**

Navigate to the linear/analysis.ipynb notebook and execute the first few cells to reproduce the results from the paper, and that's it! This will display the correlations between the regressors and human complexity ratings across all datasets. 







## Download & Unzip SVG Dataset

We further introduce a new dataset called **S**urprising **V**isual **G**enome (**SVG**) with surprising images from well-studied Visual Genome dataset along with human complexity ratings, to highlight the role of surprise in complexity judgments. We make this dataset publicly available for further research. Don't forget to cite our paper if you use this dataset in your research.

   ```bash
    # Make sure you're inside the Complexity-in-Complexity/ directory
    # Then unzip the dataset into the "SVG" folder:
    unzip SVG_dataset.zip -d SVG
   ```
