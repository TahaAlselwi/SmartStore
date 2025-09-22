# General
PROJECT_ID = ""
DATASET_ID = ""            

# BigQuery remote connection for OBJ.*
OBJ_CONNECTION_ID = f"{PROJECT_ID}.us.my_conn"

# BQ Models
TEXT_Embedding_MODEL_ID = f"{PROJECT_ID}.{DATASET_ID}.text_embedding_model"
MM_Embedding_MODEL_ID = f"{PROJECT_ID}.{DATASET_ID}.mm_embedding_model"
BQ_GENERATIVE_MODEL_ID = f"{PROJECT_ID}.{DATASET_ID}.generative_model"    
TRANSCRIPTION_MODEL_ID = f"{PROJECT_ID}.{DATASET_ID}.transcription_model"

#GCS uris for calls and images
GCS_CALLS_URI = ""
GCS_Images_URI  = ""  

## main tables
PRODUCTS_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.products"
TICKETS_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.tickets"

 







