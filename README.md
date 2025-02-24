# Insurance Complaint Analysis & Prediction

## Project Overview
This project analyzes and processes insurance complaint data using Databricks, PySpark, and machine learning techniques. The key steps include:

- **Data Cleaning and Preprocessing:** Handling missing values, converting columns to the correct data types, and feature engineering to prepare the dataset for analysis.
- **Exploratory Data Analysis (EDA):** Identifying trends and patterns in complaint resolution times, recovery amounts, and complaint statuses through visualizations.
- **Machine Learning Models:** Training and evaluating a range of models, including Random Forest, Gradient Boosting, Logistic Regression, Support Vector Classifier (SVC), and Decision Tree, to predict complaint outcomes based on various features.
- **Principal Component Analysis (PCA):** Reducing dimensionality of the dataset and visualizing relationships between key features.
- **Visualization and Reporting:** Creating visualizations from basic charts to complex ones like Sunburst charts, and assessing model performance with classification reports, including accuracy, precision, recall, and F1-score.

The entire project leverages **Databricks and PySpark** for scalable data processing, while datasets are managed using **DBFS (Databricks File System)**. Machine learning models and visualizations are created using appropriate tools and libraries, including Spark MLlib.

## Dataset
- **Dataset Name:** `Insurance_Company_Complaints_Resolutions.csv`
- **Source:** [Data.gov](https://catalog.data.gov/dataset/insurance-company-complaints-resolutions-status-and-recoveries)
- **Purpose:** Analyze and predict complaint resolution outcomes while exploring key trends in the data.

## Prerequisites
To run this project, ensure the following:

### Databricks Account
- A **Databricks Community Edition** account is sufficient.

### Cluster Runtime
- Use **Databricks Runtime 12.2 LTS** (includes Apache Spark 3.3.2, Scala 2.12).

## Required Libraries
### Data Pre-Processing
- **PySpark:** Processing large datasets and feature engineering tools like `VectorAssembler` and `PCA`.
- **Pandas:** Data manipulation and preprocessing, handling missing data, and feature encoding.
- **NumPy:** Numerical operations, including confidence interval computations.

### Data Visualization
- **Matplotlib:** Bar graphs, histograms, scatter plots.
- **Seaborn:** Heatmaps and aesthetically pleasing advanced visualizations.
- **Plotly:** Interactive visualizations like Sunburst charts to display complaint flow.

### Machine Learning
- **Scikit-Learn:** Tools and models such as Random Forest, Decision Tree, Linear Regression, Support Vector Machine, and Gradient Boosting, along with data splitting, encoding, and performance metrics.
- **Spark MLlib:** Scalable machine learning, including dimensionality reduction with PCA and models like Logistic Regression.

## How to Execute the Project
### Step-by-Step Instructions

### 1. Download and Extract Files
- Download the project zip file containing `AIT614-005-Team7.ipynb` and the dataset.
- Extract the files to a local directory for the next steps.

### 2. Log in to Databricks Community
- Access your **Databricks Community** account.

### 3. Create a Notebook
- On the left side of the interface, click on **Workspace**.
- Select **Create → Notebook** (top right).
- Click **File → Import** and upload the `.ipynb` file of the project.

### 4. Upload the Dataset
- Click **File → Upload Data to DBFS** (top left).
- Upload `dataset.csv` to **Databricks FileStore (DBFS)** and click **Next**.
- After uploading, copy the file path displayed, e.g., `dbfs:/FileStore/...`.

### 5. Update File Path in the Notebook
- Open the notebook and locate the second code cell where the dataset file path is defined.
- Replace the existing path with the copied path from Step 4.

### 6. Create a Cluster
- Click **Connect** (top right) and create a new cluster.
- Select the runtime version: **12.2 LTS (Scala 2.12, Spark 3.3.2)**.
- Wait for the cluster to initialize.

### 7. Run the Notebook
- If the cluster name appears in the top right, proceed.
- Attach the notebook to the cluster using the **top-right dropdown**.
- Click **Run All** (top-right of the notebook).
- Wait for all the cells to execute.

### 8. View Outputs
- Explore the **visualizations, insights, and results** displayed in the notebook, such as:
  - Feature importance charts
  - PCA plots
  - Model accuracy metrics

## Results and Insights
The project outputs include:
- **Visualizations** showing complaint trends (e.g., resolution times, recovery amounts).
- **Machine learning model evaluations**, including accuracy metrics.
- **Dimensionality reduction results** using PCA.

---

### 📌 *Note: This project is built for learning and research purposes. Ensure compliance with data usage policies before using real-world insurance data.*
