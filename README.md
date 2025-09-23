# 🛍️ Smart Store

**Smart Store powered by BigQuery AI** is an intelligent e-commerce platform that combines **semantic product search**, **AI-driven customer support**, and **real-time business insights** into a unified solution.

Customers can find products by meaning (via **text** or **image search**), employees can resolve tickets faster with **AI-generated replies from similar past tickets**, and managers gain a powerful dashboard that tracks **satisfaction, churn risk, and trends**.

---

## ✨ Features
- 🔍 **Semantic Product Search**: Search for products by text or image.
- 🎧 **Agent Assistant**: Employees can retrieve similar past tickets and draft replies.
- 💬 **Help Center**: Instant answers to FAQs for users.
- 📊 **Manager Dashboard**: Real-time KPIs & charts on tickets and product feedback for customer satisfaction and churn risk.
---

## 🛠️ Tools & Technologies
- **BigQuery AI** → embeddings, vector search, generative functions, and multimodal features 
- **Google Cloud Storage (GCS)** → storing product images & customer calls
- **Streamlit** → interactive multi-page web app
- **Vertex AI** → generative model for replies
- **Python (pandas, numpy, altair)** → data handling and visualization


## 📂 Project Structure
```
ٍSmartStore/
├─ .gitignore
├─ README.md                 # Project documentation
├─ requirements.txt          # Python dependencies
├─ Home.py                   # Main Streamlit entrypoint (Product Search Page)
├─ config.py                 # Configuration (Project ID, Dataset ID, Model IDs, GCS URIs)
├─ data/
│  ├─ images/                # Product images
│  ├─ calls/                 # Customer Calls
│  ├─ chats.csv              # Raw chat data
│  ├─ emails.csv             # Raw emails
│  ├─ forms.csv              # Support form submissions
│  └─ products.jsonl         # Product catalog with details/comments
├─ pages/
│  ├─ 1_Agent_Assistant.py   # Employee: find similar tickets + draft reply
│  ├─ 2_Help_Center.py       # Customer: FAQ 
│  └─ 3_Dashboard.py         # Manager: KPIs and analytics dashboards
└─ services/
   ├─ bq.py                  # BigQuery helper functions
   ├─ vertex.py              # Vertex AI services (generative)
   ├─ products_builder.py    # Build embeddings(text+image) in products table
   ├─ tickets_builder.py     # Build unified tickets table with embeddings
   └─ dashboard_helper.py    # Dashboard KPIs and charts helpers
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit toward compute/storage.
- [Make sure billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).
- [Enable the required APIs: BigQuery API, BigQuery Connection API, and Vertex AI API](https://console.cloud.google.com/apis/enableflow?apiid=bigquery.googleapis.com,bigqueryconnection.googleapis.com,aiplatform.googleapis.com).
- [Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

### 2. Authentication
```cmd
gcloud init
gcloud auth application-default login
```

### 3. BigQuery Dataset
```cmd
bq --location=US mk -d PROJECT_ID:DATASET_NAME
```

### 4. Load Data
```cmd
:: Forms
bq --location=US load --autodetect --replace --skip_leading_rows=1   --source_format=CSV DATASET_ID.forms ./data/forms.csv

:: Chats
bq --location=US load --replace --source_format=CSV DATASET_ID.chats ./data/chats.csv chat:STRING

:: Emails
bq --location=US load --replace --source_format=CSV DATASET_ID.emails ./data/emails.csv email:STRING

:: Products
bq --location=US load --autodetect --replace --source_format=NEWLINE_DELIMITED_JSON   DATASET_ID.products ./data/products.jsonl
```

### 5. Google Cloud Storage
```cmd
:: Create bucket
gsutil mb -l US -b on gs://BUCKET_NAME

:: Upload product images
gsutil -m cp -r "./data/images/*" gs://BUCKET_NAME/images/

:: Upload call audio
gsutil -m cp -r "./data/calls/*" gs://BUCKET_NAME/calls/
```

### 6. Create BigQuery Connection
```cmd
bq mk --connection --location=US --connection_type=CLOUD_RESOURCE ai_conn
```
> 
Run the following to print the connection metadata, then copy the **service account** email and use it in the IAM bindings below as `CONNECTION_SA`:

```cmd
bq show --connection PROJECT_ID.us.ai_conn
```

Grant roles to the connection’s service account:
```cmd
gcloud projects add-iam-policy-binding PROJECT_ID   --member="serviceAccount:CONNECTION_SA" --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding PROJECT_ID   --member="serviceAccount:CONNECTION_SA" --role="roles/speech.client"

gcloud projects add-iam-policy-binding PROJECT_ID   --member="serviceAccount:CONNECTION_SA" --role="roles/storage.objectViewer"
```

### 7. Create Models
```cmd
:: Text embedding model
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.text_embedding_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (ENDPOINT = 'gemini-embedding-001');"

:: Multimodal embedding model
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.mm_embedding_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (ENDPOINT = 'multimodalembedding@001');"

:: Generative model
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.generative_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (ENDPOINT = 'gemini-2.5-flash');"

:: Transcription model (Speech v2)
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.transcription_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (REMOTE_SERVICE_TYPE = 'CLOUD_AI_SPEECH_TO_TEXT_V2', SPEECH_RECOGNIZER = 'projects/PROJECT_NUMBER/locations/us/recognizers/RECOGNIZER_NAME');"
```
> **ℹ️`RECOGNIZER_NAME`**  
> You need to [**create a recognizer**](https://www.youtube.com/watch?v=Q2i0WBQxjOo) first, then copy its name into the command above where `RECOGNIZER_NAME` is.  

---

## ▶️ Run the App
```cmd
pip install -r requirements.txt
streamlit run Home.py
```
---

## 📄 License
MIT License © 2025 Smart Store
