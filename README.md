# 🛰️ Satellite Imagery-Based Property Valuation  
**Multimodal Regression using Tabular + Satellite Image Data**

---

## Overview

Real estate valuation is traditionally driven by structured attributes such as size, location, and amenities. However, **visual environmental context**—green cover, road density, proximity to water, neighborhood layout—plays a crucial role in determining property prices.

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
- Provided files:
  - `train(1).xlsx`
  - `test2.xlsx`

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
1. Load & clean tabular data
2. Fetch satellite images via API
3. Resize & normalize images
4. Align each image with its tabular record

---

### Feature Engineering
- **Tabular features:** Standard scaling & encoding
- **Visual features:** CNN-based embedding extraction
- **Fusion:** Concatenation of tabular + image embeddings

---

### Model Architecture

+------------------+      +------+      +-------------------+
|  Satellite Image | ---> | CNN  | ---> | Visual Embeddings |
+------------------+      +------+      +-------------------+
                                               |
                                               |----> +------------------+ ---> Price
                                               |      | Fully Connected  |
+------------------+      +------+      +-------------------+
| Tabular Features | ---> | MLP  | ---> | Tabular Embeddings|
+------------------+      +------+      +-------------------+

---

### Explainability
- **Grad-CAM** is applied to CNN layers
- Highlights image regions influencing price prediction
- Helps interpret:
  - Green spaces
  - Waterfront proximity
  - Road density

---

## Evaluation Metrics

| Metric | Description |
|------|------------|
| RMSE | Measures prediction error magnitude |
| R² Score | Explains variance captured by the model |

Performance comparison:
- Tabular-only model  
- Multimodal (Tabular + Image) model  

---

## Repository Structure

├── data/
│   ├── raw/
│   ├── processed/
│
├── images/
│   ├── train/
│   ├── test/
│
├── data_fetcher.py          # Satellite image downloader
├── preprocessing.ipynb     # Data cleaning & feature engineering
├── model_training.ipynb    # Multimodal model training
├── prediction.csv          # Final predictions (id, predicted_price)
├── README.md               # Project documentation

---

## Tech Stack

- **Data Handling:** Pandas, NumPy, GeoPandas  
- **Deep Learning:** PyTorch / TensorFlow  
- **Image Processing:** OpenCV, PIL  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  

---

## How to Run

Follow the steps below to clone the repository, set up the environment, and run the complete pipeline.

---

### Clone the Repository

```bash
git clone https://github.com/skyvv4lker/Satellite-Imagery-Based-Property-Valuation.git
cd Satellite-Imagery-Based-Property-Valuation
