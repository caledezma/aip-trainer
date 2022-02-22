# Build a custom training container and deploy it to AIP

Personal experimentation on using AIP when you need to use custom containers.

To build the container. Please be careful with the label, if you don't specify one, it will default to "latest"
```
docker build -f dockerfiles/Dockerfile-simple -t gcr.io/<project-name>/<container-name>:<label> ./
```

To push the container to the GCR registry.
```
docker push gcr.io/<project-name>/<container-name>:<label>
```

To run container:
```
docker run -t gcr.io/<project-name>/<container-name>:<label> --arg1 val1 --arg2 val2
```

To run with a binded folder (e.g. for saving the model in a local directory):
```
docker run -v /full/path/to/dir/in/host:/dir/in/container -t gcr.io/<project-name>/<container-name>:<label> 
```


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

You can also do the same thing as above using the script `sumbit_aip_job.py` script provided and the example training arguments file:
```
python submit_aip_job.py --training-inputs training_inputs.yaml --job-name <my_unique_job_name>
```

Bear in mind that you can modify the YAML file to suit your needs. Namely, specify the correct URI for the image that you want to run and specify the correct (or none) service account. To learn how to specify training inputs specific to the AIP job, visit the [google reference page](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput)
