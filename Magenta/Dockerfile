FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the source code (assuming 'run_dashboard.py' is in the 'source' directory)
COPY . /app

# Set the environment variables for the conda environment
ENV PATH /opt/conda/envs/magenta_env/bin:$PATH

# Define the command to run your script
CMD ["/opt/conda/envs/magenta_env/bin/python", "source/run_dashboard.py"]

