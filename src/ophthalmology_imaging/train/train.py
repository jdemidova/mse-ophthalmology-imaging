# Example of using config in training

# from ophthalmology_imaging.tracking.mlflow_tracker import log_config, log_metrics, start_run
#
# config = {
#     "model": {"name": "unet", "encoder": "resnet34"},
#     "train": {"lr": 1e-4, "batch_size": 8},
# }
#
# with start_run(
#     experiment_name="semantic-segmentation",
#     run_name="debug-run",
#     tracking_uri="http://localhost:5002",
# ):
#     log_config(config)
#     log_metrics({"train/loss": 0.7, "val/dice": 0.62}, step=1)