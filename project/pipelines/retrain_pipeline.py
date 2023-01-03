from zenml.pipelines import pipeline

from utils.config_handler import get_main_config

config = get_main_config()


@pipeline(enable_cache=False, name="retrain_pipeline_v1")
def training_model(
        download_data,
        process_data,
        split_data,
        oversampling,
        train_model,
        get_mlflow_model,
        predict_new_model,
        predict_current_model,
        evaluate_new_model,
        evaluate_current_model,
        log_metrics,
        rotate_models
):
    # Download data from BigQuery
    data = download_data()
    # Process data
    processed_data = process_data(data=data)
    # Split in train/test dataframes
    X_train, X_test, y_train, y_test = split_data(data=processed_data)
    # Apply data_oversampler
    X_train_res, y_train_res = oversampling(X_train=X_train, y_train=y_train)
    # Train the old_models
    model_new = train_model(X_train_res=X_train_res, y_train_res=y_train_res)
    model_current = get_mlflow_model()
    # Get predictions on test dataset
    y_pred_new, y_pred_proba_new = predict_new_model(model=model_new, X=X_test)
    y_pred_current, y_pred_proba_current = predict_current_model(model=model_current, X=X_test)
    # Evaluate the old_models
    metrics_new = evaluate_new_model(y_pred=y_pred_new, y_pred_proba=y_pred_proba_new, y_test=y_test)
    metrics_current = evaluate_current_model(y_pred=y_pred_current, y_pred_proba=y_pred_proba_current, y_test=y_test)
    log_metrics(metrics=metrics_new)
    # Check if new model outperforms current model
    rotate_models(metrics_new=metrics_new, metrics_current=metrics_current)
