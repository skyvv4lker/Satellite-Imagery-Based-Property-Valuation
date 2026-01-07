# 🛰️ Satellite Imagery-Based Property Valuation  
**Multimodal Regression using Tabular + Satellite Image Data**

---

## Overview

Real estate valuation is traditionally driven by structured attributes such as size, location, and amenities. However, **visual environmental context**—green cover, road density, proximity to water, and neighborhood layout—plays a crucial role in determining property prices.

This project builds a **multimodal regression pipeline** that predicts **property market value** by combining:

- **Tabular housing data** (numerical & categorical features)
- **Satellite imagery** fetched programmatically using latitude & longitude

By integrating both data types, the model captures not just *what the house is*, but *where and how it is situated*.

---

## Objectives

- Build a **multimodal regression model** for house price prediction  
- Programmatically fetch **satellite images** using geo-coordinates  
- Perform **EDA & geospatial analysis** on housing and visual data  
- Extract **deep visual features** using CNNs  
- Experiment with **fusion strategies** for tabular + image data  
- Apply **model explainability (Grad-CAM)** to understand visual influence  
- Compare performance of **tabular-only vs multimodal models**

---

## Dataset

### Tabular Data

**Source:**  
- Kaggle House Sales Dataset  

**Files Used (after conversion & preprocessing):**
- `datasets/train_data.csv`
- `datasets/test_data.csv`
- `datasets/train_preprocessed.csv`
- `datasets/test_preprocessed.csv`

**Key Features**

| Feature | Description |
|------|------------|
| `price` | Target variable |
| `bedrooms`, `bathrooms` | Basic house attributes |
| `sqft_living` | Total living area |
| `sqft_above`, `sqft_basement` | Above & below ground area |
| `sqft_lot` | Total land area |
| `sqft_living15`, `sqft_lot15` | Neighborhood density indicators |
| `condition` | Maintenance quality (1–5) |
| `grade` | Construction & design quality (1–13) |
| `view` | View quality rating (0–4) |
| `waterfront` | Waterfront indicator (0/1) |
| `lat`, `long` | Geographic coordinates |

---

### Satellite Image Data

- Images are fetched using **latitude & longitude**
- Captures:
  - Green cover  
  - Road connectivity  
  - Water bodies  
  - Urban density  

**Supported APIs**
- Google Maps Static API  
- Mapbox Static Images API  
- Sentinel Hub  

---

## Methodology

### Data Pipeline

1. Load raw CSV data from `datasets/`
2. Perform EDA and preprocessing
3. Fetch satellite images using coordinates
4. Normalize and align images with tabular records
5. Train multimodal regression model
6. Generate final predictions

---

### Feature Engineering

- **Tabular features:** Scaling & encoding (performed in preprocessing notebook)
- **Visual features:** CNN-based embedding extraction
- **Fusion:** Concatenation of tabular + image embeddings

---

### Model Architecture

```text
+------------------+      +------+      +-------------------+
|  Satellite Image | ---> | CNN  | ---> | Visual Embeddings |
+------------------+      +------+      +-------------------+
                                               |
                                               |----> +------------------+ ---> Price
                                               |      | Fully Connected  |
+------------------+      +------+      +-------------------+
| Tabular Features | ---> | MLP  | ---> | Tabular Embeddings|
+------------------+      +------+      +-------------------+
```

### Repository Structure

```text
├── EDA/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│
├── data_fetcher/
│   ├── data_fetcher.ipynb
│
├── datasets/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── train_preprocessed.csv
│   ├── test_preprocessed.csv
│
├── model_training/
│   ├── multimodal_training_testing.ipynb
|   ├── XGB_tabular.ipynb
│
├── 22118009_final.csv
├── README.md
├── LICENSE
```


