import wandb
run = wandb.init()
artifact = run.use_artifact('dohoangvunt2005/california_housing_regression/run-bv9oragx-history:v0', type='wandb-history')
artifact_dir = artifact.download()