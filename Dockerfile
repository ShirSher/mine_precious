# Define base image
FROM nvcr.io/nvidia/pytorch:19.07-py3
 
# Set working directory for the project
WORKDIR /Syclop-MINE

COPY . .

CMD ["python3", "DriftMINERun.py"]