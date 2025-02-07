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
3. **Surprise Scores** derived from **Large Language Models**, indicating **unusual/novel** objects or contexts.

Using these features alongside existing segmentation/object-based features, we demonstrate improved performance in predicting **human-rated visual complexity** across multiple datasets. We further introduce a new dataset called **S**urprising **V**isual **G**enome (**SVG**) with surprising images from well-studied Visual Genome dataset, to highlight the role of surprise in complexity judgments.



---

## Quick Start

1. **Clone and Install**

   ```bash
   git clone https://github.com/Complexity-Project/Complexity-in-Complexity.git
   cd Complexity-in-Complexity
   pip install -r requirements.txt
   ```
2. **Download & Unzip SVG Dataset**

   ```bash
    # Make sure you'reinside the Complexity-in-Complexity/ directory
    # Then unzip the dataset into an "SVG" folder:
    unzip SVG_dataset.zip -d SVG
   ```