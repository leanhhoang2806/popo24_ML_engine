# Search for a specific version of tensorflow depends on your GPU
FROM tensorflow/tensorflow:latest-gpu


# Set the working directory inside the container
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip setuptools
# RUN apt-get install -y --no-install-recommends \
#     openssh-client 

# Copy the requirements.txt file to the container's working directory
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application files
COPY . .


# Set the entry point to run main.py when the container starts
CMD ["python", "-m", "single_training"]