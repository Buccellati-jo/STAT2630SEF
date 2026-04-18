# STAT2630SEF Course Project: Analysis of YouTube Shorts View Count Drivers
This is the full code repository for my STAT2630SEF course project, which implements a complete distributed data analysis pipeline for YouTube Shorts view count factor analysis, built with PySpark as the core framework.

## Project Overview
This project covers the full workflow of short video analytics, from compliant data collection, comment sentiment analysis, to distributed data preprocessing, feature engineering, and regression modeling. The core goal is to explore the key factors that influence YouTube Shorts view counts, and meet the course's requirements for distributed computing-based data analysis.

## Code File Structure
### 1. Data Collection & Preprocessing Pipeline
| File Name | Function |
|-----------|----------|
| `config.py` | Centralized configuration file for YouTube API, MongoDB connection, and global parameters |
| `crawl_youtube.py` | Compliant data crawling via the official YouTube Data API, collects video metadata, engagement data, and user comments |
| `data_cleaning.py` | Data cleaning workflow, including deduplication, missing value handling, and outlier filtering |
| `feature_engineering.py` | Multi-dimensional feature construction, including time features, text features, and categorical feature encoding |
| `sentiment_analysis.py` | Comment sentiment analysis with the VADER model, calculates video-level average sentiment scores |

### 2. PySpark Distributed Analysis Pipeline (Core Course Requirement)
| File Name | Function |
|-----------|----------|
| `01_data_check.py` | Data loading, preprocessing, feature engineering, and data preview with PySpark |
| `02_descriptive_analysis.py` | Descriptive statistics of core variables, implemented natively in PySpark |
| `03_correlation_analysis.py` | Pearson correlation analysis between features and view counts, with PySpark |
| `04_linear_regression.py` | Multiple linear regression model construction and evaluation, with PySpark MLlib |
| `05_random_forest.py` | Random forest regression model construction and feature importance ranking, with PySpark MLlib |

## Environment & Dependencies
- Python 3.9+
- PySpark 3.3.0+
- MongoDB 5.0+
- NLTK (VADER sentiment model)
- Google API Python Client (for YouTube Data API)
- Pandas, Scikit-learn

## How to Run
1.  Configure your YouTube API key and MongoDB connection in `config.py`
2.  Run `crawl_youtube.py` to collect raw data
3.  Run `sentiment_analysis.py` to calculate comment sentiment scores
4.  Run `data_cleaning.py` and `feature_engineering.py` to prepare modeling data
5.  Run the PySpark analysis files in order from 01 to 05 to reproduce the full analysis results

## Project Conclusion
The core finding of this project is that user engagement (especially the number of likes) is the strongest driver of YouTube Shorts view counts. The linear regression model built with PySpark achieves an R² of 0.9387 on the test set, which can reliably predict video view counts.
