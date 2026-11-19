# Vultr GPU runbook

This workflow uses a manually provisioned, on-demand NVIDIA GPU. The previous training run used an A10G with 24 GB VRAM; GPU availability, software compatibility, and current pricing must be checked before committing to a plan.

## Before provisioning

- Identify the dataset and confirm that it is permitted to be processed on Vultr. Remove secrets and sensitive information as appropriate.
- Agree on the region and a spending ceiling for both the training run and inference usage. Inspect the current Vultr plan's hourly rate, availability, storage costs, and expected runtime in the Vultr console.
- Prepare an SSH key, artifact backup destination, and Hugging Face credentials with access to the base model and the intended destination. Do not commit tokens, datasets containing private logs, or API keys to the repository.
- Choose a GPU image with a compatible NVIDIA driver, CUDA runtime, and Python version. Restrict SSH and API access using a firewall. Use an SSH tunnel or TLS termination for remote API access.

## Provision and prepare

1. In the Vultr console, create an NVIDIA Cloud GPU instance in the agreed region. Select a GPU with sufficient VRAM, disk space for the base model, checkpoints, data, and virtual environment, and add the SSH key during provisioning. Record the instance ID and creation time.
2. Connect over SSH and check `nvidia-smi`, free disk space, Python, and access to the dataset and artifact destination. Verify the installed CUDA/driver stack is compatible with the selected dependencies.
3. Clone the repository and check out the intended release or commit. Create a virtual environment and install the project dependencies and record the resolved versions for the run. GPU training dependencies are not yet pinned.
4. Copy the approved dataset to the server over an encrypted channel. Validate it and create the training, validation, and held-out test splits using the release's documented commands. Keep test data out of training and model selection.
5. Authenticate to Hugging Face on the instance using the supported login command. Confirm read access to the base model and write access to a **new** destination before training. Do not put credentials in command history or artifacts.

## Train and evaluate

1. Run the documented short GPU smoke test and verify that it creates a loadable adapter and usable metrics.
2. Run the full training command with a unique run ID and output directory. Monitor GPU utilization, error logs, elapsed time, and accrued cost against the agreed ceiling. Stop the run if the ceiling or a failure condition is reached.
3. Evaluate the existing adapter and retrained adapter on the same held-out test set. Record JSON validity, severity performance, and a human review of diagnosis quality. Save the exact model revisions, dataset split identifiers, configuration, logs, and metrics.
4. Verify the saved adapter by loading it with its base model and generating a response. Publish the new version only after checking that artifacts contain no secrets or private logs and the target does not overwrite the existing adapter.

## Serve and verify

Start the release's inference API bound to localhost, with a secret API token loaded from the environment or a protected secret file. Expose it through an SSH tunnel, private network, or properly configured TLS reverse proxy. Do not expose an unauthenticated development server to the internet.

Run the documented authenticated health and inference smoke tests with a non-sensitive sample. Confirm real model generation, schema validation, the model revision, latency, and failure responses. Model-generated confidence is not a calibrated probability; diagnoses require human review.

## Backup and teardown

1. Copy the adapter, tokenizer, run configuration, evaluation report, and required logs to the approved durable destination. Verify that the backup can be read and loaded. Avoid retaining raw sensitive logs unnecessarily.
2. Shut down the application and **destroy the instance** in the Vultr console when finished. Confirm that the instance is gone and separately review persistent volumes, snapshots, reserved IPs, and other chargeable resources.
3. Verify the final bill and record the elapsed time, GPU type, and actual cost. Recreate an instance from the documented configuration and restore the verified adapter when service is needed again.

A stopped Vultr instance continues to incur charges; destroying it releases the instance and permanently deletes its local data. See [Vultr's billing documentation](https://docs.vultr.com/support/platform/billing/are-stopped-instances-still-billed-on-vultr).
