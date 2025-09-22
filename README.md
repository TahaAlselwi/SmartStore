# Smart Store

The AI-Powered E-Commerce Platform with BigQuery AI

---

## Overview

Smart Store powered by BigQuery AI is an intelligent e-commerce platform that combines semantic product search, AI-driven customer support, and real-time business insights into a unified solution. Customers can find products by meaning (via text or image search), employees can resolve tickets faster with AI-generated replies and priority detection, and managers gain a powerful dashboard that tracks satisfaction, churn risk, and trends.

By leveraging BigQuery AI functions such as ML.GENERATE_EMBEDDING, AI.GENERATE, and AI.GENERATE_BOOL, the system transforms unstructured data (reviews, chats, emails, calls) into actionable intelligence. This results in smarter shopping experiences, more efficient support teams, and data-driven decision making for business leaders.

In short, Smart Store bridges the gap between raw data and real customer value, showing how BigQuery AI can power the next generation of digital commerce.

---

## Prerequisites

- [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a \$300 free credit toward compute/storage.
- [Make sure billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).
- [Enable the required APIs: BigQuery API, BigQuery Connection API, and Vertex AI API](https://console.cloud.google.com/apis/enableflow?apiid=bigquery.googleapis.com,bigqueryconnection.googleapis.com,aiplatform.googleapis.com).
- [Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

## Quick Setup
### Project Setup

```cmd
gcloud init
```

> Logs you into Google Cloud and sets your default project/config for subsequent commands.

### Application Default Credentials

```cmd
gcloud auth application-default login
```

> Provides Application Default Credentials (ADC) for CLI tools and client libraries.

### Create BigQuery Dataset

```cmd
bq --location=US mk -d PROJECT_ID:DATASET_NAME
```

> Creates the BigQuery dataset where tables and models will live.

## Loading Data
### Load forms.csv

```cmd
bq --location=US load --autodetect --replace --skip_leading_rows=1 --source_format=CSV DATASET_ID.forms ./forms.csv
```

> Loads forms.csv into DATASET_ID.forms with autodetected schema and header skipped.

### Load chats.csv

```cmd
bq --location=US load --replace --source_format=CSV DATASET_ID.chats ./chats.csv chat:STRING
```

> Loads chats.csv into DATASET_ID.chats with a single-column schema (chat\:STRING).

### Load emails.csv

```cmd
bq --location=US load --replace --source_format=CSV DATASET_ID.emails ./emails.csv email:STRING
```

> Loads emails.csv into DATASET_ID.emails with a single-column schema (email\:STRING).

### Load products.jsonl

```cmd
bq --location=US load --autodetect --replace --source_format=NEWLINE_DELIMITED_JSON DATASET_ID.products ./products.jsonl
```

> Loads products.jsonl into DATASET\_ID.products using JSON Lines autodetection.

### Create GCS bucket (US)

```cmd
gsutil mb -l US -b on gs://BUCKET_NAME
```

> Creates a regional GCS bucket to store images and audio.

### Upload product images

```cmd
gsutil -m cp -r "<LOCAL_IMAGES_FOLDER>/*" gs://BUCKET_NAME/images/
```

> Uploads product images to GCS.

### Upload call audio files

```cmd
gsutil -m cp -r "<LOCAL_CALLS_FOLDER>/*" gs://BUCKET_NAME/calls/
```

> Uploads call audio files to GCS.

## Creating a Connection and Granting Roles
### Create unified BigQuery connection

```cmd
bq mk --connection --location=US --connection_type=CLOUD_RESOURCE ai_conn
```

> Creates a reusable BigQuery connection for all remote models.

### Get the connection’s managed service account

```cmd
bq show --connection PROJECT_ID.us.ai_conn
```

> Displays the connection metadata to copy the managed service account.

### Grant Vertex AI User role (project-level)

```cmd
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:CONNECTION_SA" --role="roles/aiplatform.user" --condition=None
```

> Lets the connection call Vertex AI endpoints.

### Grant Speech Client role (project-level)

```cmd
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:CONNECTION_SA" --role="roles/speech.client" --condition=None
```

> Allows using Speech-to-Text v2.

### Grant Storage Object Viewer role (project-level)

```cmd
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:CONNECTION_SA" --role="roles/storage.objectViewer" --condition=None
```

> Grants read access to GCS objects in the project.

## Creating Models
### Create text embedding model

```cmd
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.text_embedding_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (ENDPOINT = 'gemini-embedding-001');"
```

> Creates a remote text embedding model (Gemini) for ML.GENERATE\_EMBEDDING.

### Create multimodal embedding model

```cmd
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.mm_embedding_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (ENDPOINT = 'multimodalembedding@001');"
```

> Creates a remote multimodal embedding model for image/text embeddings.

### Create generative model (Gemini 2.5 Flash)

```cmd
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.generative_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (ENDPOINT = 'gemini-2.5-flash');"
```

> Creates a remote generative model for AI.GENERATE\_TABLE.

### Create transcription model (Speech v2) — requires recognizer

```cmd
bq query --use_legacy_sql=false "CREATE OR REPLACE MODEL `PROJECT_ID.DATASET_ID.transcription_model` REMOTE WITH CONNECTION `PROJECT_ID.us.ai_conn` OPTIONS (REMOTE_SERVICE_TYPE = 'CLOUD_AI_SPEECH_TO_TEXT_V2', SPEECH_RECOGNIZER = 'projects/PROJECT_NUMBER/locations/us/recognizers/RECOGNIZER_NAME');"
```

> Creates a remote transcription model bound to your recognizer. ([Create Speech Recognizer](https://www.youtube.com/watch?v=Q2i0WBQxjOo))
