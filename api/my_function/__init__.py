import azure.functions as func
import logging
import os
from azure.storage.blob import BlobServiceClient
import joblib
import json
import scipy.sparse as sp
import numpy as np

model = None
user2idx = None
article2idx = None
user_item_matrix = None

def load_model_from_blob():
    global model, user2idx, article2idx, user_item_matrix

    logging.info("Loading model and artifacts from Azure Blob Storage...")
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connect_str:
        logging.error("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")
        return None, None, None, None

    container_name = "models"

    try:
        # Initialize BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)

        # Download and load the ALS model
        model_blob_client = container_client.get_blob_client("model.pkl")
        with open("model.pkl", "wb") as download_file:
            download_file.write(model_blob_client.download_blob().readall())
        model = joblib.load("model.pkl")
        logging.info("Model loaded successfully.")

        # Download and load user factors
        user_factors_blob_client = container_client.get_blob_client("user_factors.npy")
        with open("user_factors.npy", "wb") as download_file:
            download_file.write(user_factors_blob_client.download_blob().readall())
        model.user_factors = np.load("user_factors.npy")
        logging.info("User factors loaded successfully.")

        # Download and load item factors
        item_factors_blob_client = container_client.get_blob_client("item_factors.npy")
        with open("item_factors.npy", "wb") as download_file:
            download_file.write(item_factors_blob_client.download_blob().readall())
        model.item_factors = np.load("item_factors.npy")
        logging.info("Item factors loaded successfully.")

        # Download and load artifacts
        artifacts_blob_client = container_client.get_blob_client("artifacts.json")
        with open("artifacts.json", "wb") as download_file:
            download_file.write(artifacts_blob_client.download_blob().readall())
        with open("artifacts.json", "r") as f:
            artifacts = json.load(f)
        logging.info("Artifacts loaded successfully.")

        # Extract user2idx and article2idx mappings
        user2idx = artifacts.get("user2idx", {})
        article2idx = artifacts.get("article2idx", {})
        if not user2idx or not article2idx:
            logging.warning("User or article indices are missing in artifacts.")

        # Reconstruct the user-item matrix
        user_item_matrix = sp.coo_matrix(
            (artifacts["user_item_matrix"]["data"],
             (artifacts["user_item_matrix"]["row"],
              artifacts["user_item_matrix"]["col"])),
            shape=artifacts["user_item_matrix"]["shape"]
        ).tocsr()
        logging.info("User-item matrix successfully reconstructed.")

        return model, user2idx, article2idx, user_item_matrix

    except Exception as e:
        logging.error(f"Failed to load model and artifacts: {e}")
        return None, None, None, None

def get_popular_articles(user_item_matrix, article2idx, num_articles):
    article_popularity = user_item_matrix.sum(axis=0).A1
    
    idx2article = {idx: int(article) for article, idx in article2idx.items()}
    
    popular_article_indices = np.argsort(-article_popularity)
    
    popular_articles = [
        idx2article[idx]
        for idx in popular_article_indices[:num_articles]
        if idx in idx2article
    ]
    
    return popular_articles

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
    
    if len(recommended_articles) < n_items:
        popular_articles = get_popular_articles(user_item_matrix, article2idx, n_items - len(recommended_articles))
        recommended_articles.extend(popular_articles)

    return recommended_articles[:n_items]

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.INFO)
    logging.info('Python HTTP trigger function processed a request.')
    

    user_id = req.params.get('user_id')
    if not user_id:
        return func.HttpResponse(
            "Please pass a user_id on the query string.",
            status_code=400
        )

    global model, user2idx, article2idx, user_item_matrix
    if model is None:
        logging.info("Model is not loaded. Attempting to load from blob storage...")
        model, user2idx, article2idx, user_item_matrix = load_model_from_blob()
        if model is None:
            logging.error("Failed to load model from blob storage.")
            return func.HttpResponse(
                "Error loading model. Please check logs for details.",
                status_code=500
            )

    # Get recommendations
    recommended_articles = get_cf_recommendations(user_id, model, user2idx, article2idx, user_item_matrix)
    logging.info(f"Recommended articles for user {user_id}: {recommended_articles}")

    return func.HttpResponse(json.dumps(recommended_articles), mimetype="application/json")