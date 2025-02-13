import azure.functions as func
import logging
import os
from azure.storage.blob import BlobServiceClient
import joblib
import json
import scipy.sparse as sp

model = None
user2idx = None
article2idx = None
user_item_matrix = None

def load_model_from_blob():
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "models"

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    # Download the model
    model_blob_client = container_client.get_blob_client("model.pkl")
    with open("model.pkl", "wb") as download_file:
        download_file.write(model_blob_client.download_blob().readall())
    model = joblib.load("model.pkl")

    # Download the artifacts
    artifacts_blob_client = container_client.get_blob_client("artifacts.json")
    with open("artifacts.json", "wb") as download_file:
        download_file.write(artifacts_blob_client.download_blob().readall())
    with open("artifacts.json", "r") as f:
        artifacts = json.load(f)
    
    user2idx = artifacts["user2idx"]
    article2idx = artifacts["article2idx"]
    user_item_matrix = sp.coo_matrix(
        (artifacts["user_item_matrix"]["data"],
         (artifacts["user_item_matrix"]["row"],
          artifacts["user_item_matrix"]["col"])),
        shape=artifacts["user_item_matrix"]["shape"]
    ).tocsr()

    return model, user2idx, article2idx, user_item_matrix

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="product_get")
def product_get(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('user_id')
    if not user_id:
        return func.HttpResponse(
            "Please pass a user_id on the query string.",
            status_code=400
        )

    global model, user2idx, article2idx, user_item_matrix
    if model is None:
        model, user2idx, article2idx, user_item_matrix = load_model_from_blob()

    # Get recommendations
    recommended_articles = get_cf_recommendations(user_id, model, user2idx, article2idx, user_item_matrix)

    return func.HttpResponse(json.dumps(recommended_articles), mimetype="application/json")

def get_cf_recommendations(user_id, model, user2idx, article2idx, user_item_matrix, n_items=5):
    user_id = int(user_id)
    if user_id not in user2idx:
        return []
    user_idx = user2idx[user_id]
    item_ids, scores = model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=n_items,
        filter_already_liked_items=True
    )
    idx2article = {idx: int(article) for article, idx in article2idx.items()}
    recommended_articles = [
        idx2article[int(idx)]
        for idx in item_ids
        if int(idx) in idx2article
    ]
    return recommended_articles