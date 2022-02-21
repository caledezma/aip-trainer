# Build a custom training container and deploy it to AIP

Personal experimentation on using AIP when you need to use custom containers.

To submit a GPU job:

```
gcloud ai-platform jobs submit training $JOB_NAME \
  --scale-tier BASIC_GPU \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --epochs=5 \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR
```

To build the container
```
docker build -f dockerfiles/Dockerfile-simple -t gcr.io/<project-name>/<container-name>:<label> ./
```

To push the container
```
docker push gcr.io/<project-name>/<container-name>:<label>
```
