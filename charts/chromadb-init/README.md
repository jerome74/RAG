# chromadb-init Helm Chart

This chart deploys a one-off Kubernetes Job that builds a ChromaDB persistent collection using OpenAI embeddings and stores the database on a PersistentVolumeClaim (PVC).

## Contents
- Overview
- Prerequisites
- Build the container image
- Provide the CSV data
- Install with Helm
- Verify and inspect
- Re-run the initializer
- Configuration reference
- Security notes
- Data output

## Overview
- The initializer runs [apps/chromadb_init/main.py](../../apps/chromadb_init/main.py) inside a container built from [apps/chromadb_init/Dockerfile](../../apps/chromadb_init/Dockerfile).
- It writes a persistent ChromaDB to a volume mounted at `/data` (configurable), defaulting to `/data/vectordb` in [charts/chromadb-init/values.yaml](values.yaml).
- OpenAI API key is provided via a Kubernetes Secret rendered from [charts/chromadb-init/templates/secret.yaml](templates/secret.yaml).
- Storage is provided by a PVC rendered from [charts/chromadb-init/templates/pvc.yaml](templates/pvc.yaml) or you can plug an existing one.
- The Job definition is at [charts/chromadb-init/templates/job.yaml](templates/job.yaml).

## Prerequisites
- Docker or another OCI-compliant builder
- Access to a container registry (e.g., GHCR, Docker Hub)
- A Kubernetes cluster and `kubectl` context
- Helm 3.9+
- An OpenAI API key with access to the specified embedding model

## Build the container image
From the repository root:

    docker build -t your-docker-repo/chromadb-init:0.1.0 -f apps/chromadb_init/Dockerfile .
    docker push your-docker-repo/chromadb-init:0.1.0

The image runs the initializer with the entrypoint defined in [apps/chromadb_init/Dockerfile](../../apps/chromadb_init/Dockerfile).

## Provide the CSV data
You have two options:

1) Download via URL (recommended for large files)
   - Host your CSV at an accessible URL (HTTPS recommended).
   - Set `values.csv.url` to that URL.

2) Bake CSV into the image
   - Copy your CSV into the image at `/work/knowledgebase_for_chromadb.csv` or another path, and set `values.csv.path` accordingly.
   - Example Dockerfile change:

        # in apps/chromadb_init/Dockerfile (optional)
        # COPY ../../archive/knowledgebase_for_chromadb.csv /work/knowledgebase_for_chromadb.csv

   - Note: If you use option 2, ensure you build with repository root context (`.`) so the `archive` file is in scope.

## Install with Helm
Set your image repository/tag and supply the OpenAI API key securely. Example using a local file:

    echo -n "sk-your-openai-key" > openai.key
    helm upgrade --install chromadb-init charts/chromadb-init \
      --set image.repository=your-docker-repo/chromadb-init \
      --set image.tag=0.1.0 \
      --set-file secret.openaiApiKey=./openai.key \
      --set csv.url="https://example.com/knowledgebase_for_chromadb.csv"

If you want to use an existing PVC instead of creating one:

    helm upgrade --install chromadb-init charts/chromadb-init \
      --set image.repository=your-docker-repo/chromadb-init \
      --set image.tag=0.1.0 \
      --set-file secret.openaiApiKey=./openai.key \
      --set pvc.create=false \
      --set pvc.existingClaim=your-existing-claim

## Verify and inspect
- Check Job and Pod:

      kubectl get jobs
      kubectl get pods -l app.kubernetes.io/instance=chromadb-init
      kubectl logs job/chromadb-init-init

- Confirm the PVC:

      kubectl get pvc

The script logs include the number of rows ingested and the final collection count.

## Re-run the initializer
Jobs are not restarted automatically. To run again:

1) Update a value (e.g., image.tag), then:

      helm upgrade chromadb-init charts/chromadb-init --set image.tag=0.1.1

or

2) Uninstall and reinstall:

      helm uninstall chromadb-init
      helm install chromadb-init charts/chromadb-init \
        --set image.repository=your-docker-repo/chromadb-init \
        --set image.tag=0.1.0 \
        --set-file secret.openaiApiKey=./openai.key \
        --set csv.url="https://example.com/knowledgebase_for_chromadb.csv"

## Configuration reference
Main settings in [charts/chromadb-init/values.yaml](values.yaml):

- image.repository: container image repository
- image.tag: image tag
- image.pullPolicy: IfNotPresent|Always
- chromadb.collectionName: name of the collection (default: `derms_kb`)
- chromadb.dbPath: mount path inside pod, default `/data/vectordb`
- chromadb.modelName: OpenAI model (default: `text-embedding-3-small`)
- chromadb.batchSize: embedding batch size (default: `128`)
- csv.url: URL to download CSV (takes precedence over `csv.path`)
- csv.path: in-container path to the CSV if not using URL
- csv.encoding: file encoding (default `ISO-8859-1`)
- csv.delimiter: CSV delimiter, default `,`
- csv.failOnMissingColumns: boolean; fail if `"document"` column absent
- pvc.create: whether to create a new PVC
- pvc.name: name of the created PVC
- pvc.existingClaim: use an existing PVC instead of creating one
- pvc.mountPath: where to mount the PVC (default: `/data`)
- pvc.storageClassName: storage class to use (optional)
- pvc.accessModes: defaults to `["ReadWriteOnce"]`
- pvc.size: requested storage size
- secret.create: whether to create the Secret
- secret.name: Secret name override
- secret.openaiApiKey: OpenAI API key content (use `--set-file`)
- job.backoffLimit: Job retries (default: `1`)
- job.ttlSecondsAfterFinished: TTL after completion
- job.restartPolicy: `Never` by default
- resources, nodeSelector, tolerations, affinity: standard pod settings

## Security notes
- Do not commit your OpenAI API key to source control.
- Prefer passing the key with `--set-file` as shown above.
- Consider tightening Pod `securityContext` and using an `imagePullSecret` if your registry is private.
- Rotate the API key that appears in the original notebook; it should be treated as compromised.

## Data output
- The resulting ChromaDB is persisted under the mounted PVC at the configured `chromadb.dbPath` (default `/data/vectordb`). You can later mount the same PVC into your ChromaDB-serving deployment or any consumer that requires read access.