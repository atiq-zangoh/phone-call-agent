# Use an official lightweight Python image.
FROM python:3.10

# Set a working directory inside the container.
WORKDIR /app

# Install system dependencies (if needed for building some packages).
RUN apt-get update && apt-get install -y build-essential

# Copy dependency list and install Python dependencies.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Set the default command to run your phone call agent.
CMD ["python", "phone_rag.py", "start"]
