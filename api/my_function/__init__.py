import azure.functions as func
import logging
import os
import tempfile
from azure.storage.blob import BlobServiceClient
import json
import scipy.sparse as sp
import numpy as np
from implicit.als import AlternatingLeastSquares

# Global storage variables for model and artifacts
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
        if not container_client.exists():
            logging.error(f"Container '{container_name}' does not exist.")
            return None, None, None, None
        
        logging.info("Connected to Azure Blob Storage.")

        # Temporary directory using tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, "cf_model.npz")
            temp_usrmatrix_path = os.path.join(temp_dir, "user_item_matrix.npz")
            temp_artifacts_path = os.path.join(temp_dir, "artifacts.json")

            logging.info(f"Temporary directory created at: {temp_dir}")

            # Download and load the ALS model (from cf_model.npz)
            model_blob_client = container_client.get_blob_client("cf_model.npz")
            with open(temp_model_path, "wb") as download_file:
                download_file.write(model_blob_client.download_blob().readall())
            logging.info("Model downloaded successfully.")

            # Load the ALS model's user and item factors
            model_data = np.load(temp_model_path)
            user_factors = model_data['user_factors']
            item_factors = model_data['item_factors']
            
            # Reconstruct the model (Implicit ALS)
            model = AlternatingLeastSquares(factors=user_factors.shape[1])
            model.user_factors = user_factors
            model.item_factors = item_factors
            logging.info("Model loaded successfully.")

            # Download and load the user-item matrix (from user_item_matrix.npz)
            matrix_blob_client = container_client.get_blob_client("user_item_matrix.npz")
            with open(temp_usrmatrix_path, "wb") as download_file:
                download_file.write(matrix_blob_client.download_blob().readall())

            # Load the user-item matrix from the .npz file
            user_item_matrix = sp.load_npz(temp_usrmatrix_path)
            logging.info("User-item matrix loaded successfully.")

            # Download and load artifacts (from artifacts.json)
            artifacts_blob_client = container_client.get_blob_client("artifacts.json")
            with open(temp_artifacts_path, "wb") as download_file:
                download_file.write(artifacts_blob_client.download_blob().readall())
            with open(temp_artifacts_path, "r") as f:
                artifacts = json.load(f)
            logging.info("Artifacts loaded successfully.")

            # Extract user2idx and article2idx mappings
            user2idx = artifacts.get("user2idx", {})
            article2idx = artifacts.get("article2idx", {})
            if not user2idx or not article2idx:
                logging.warning("User or article indices are missing in artifacts.")

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
    user_id = str(user_id)
    if user_id not in user2idx:
        logging.warning(f"User ID {user_id} not found in user2idx.")
        return []
    
    user_idx = user2idx[user_id]
    logging.info(f"Generating recommendations for user {user_id} (user index: {user_idx})...")
    
    user_data = user_item_matrix[user_idx]
    logging.info(f"user_item_matrix[user_idx]: {user_data.toarray() if user_data.nnz else 'No interactions'}")

    item_ids, scores = model.recommend(
        user_idx,
        user_data,
        N=n_items,
        filter_already_liked_items=True
    )
    logging.info(f"Model recommendations: {item_ids}, Scores: {scores}")
    
    idx2article = {idx: str(article) for article, idx in article2idx.items()}
    recommended_articles = [
        idx2article.get(idx, None)
        for idx in item_ids
        if idx in idx2article
    ]
    
    logging.info(f"Recommended articles: {recommended_articles}")
    
    if len(recommended_articles) < n_items:
        logging.info(f"Not enough recommendations. Adding popular articles.")
        popular_articles = get_popular_articles(user_item_matrix, article2idx, n_items - len(recommended_articles))
        recommended_articles.extend(popular_articles)
    
    logging.info(f"Final recommended articles: {recommended_articles}")
    return recommended_articles[:n_items]

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Python HTTP trigger function processed a request.')
    

    user_id = req.params.get('user_id')
    if not user_id:
        return func.HttpResponse(
            "Please pass a user_id on the query string.",
            status_code=400
        )

    # Load model and artifacts if not already loaded
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