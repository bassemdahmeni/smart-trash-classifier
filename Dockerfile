FROM continuumio/miniconda3

WORKDIR /app

# Copy your pip requirements
COPY requirements.txt .

# Create the conda environment
RUN conda create -y --name myenv python=3.10

# Install pip packages inside that conda env
RUN conda run -n myenv pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Make conda env the default for all RUN/CMD
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Run your Flask app
CMD ["python", "app.py"]
