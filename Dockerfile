FROM python:3.10-slim

# Install ttyd for web terminal
RUN apt-get update && apt-get install -y ttyd && rm -rf /var/lib/apt/lists/*

# Create data directory (standard Unix location)
RUN mkdir -p /usr/local/share/axono

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install --no-cache-dir -e .

# Expose ttyd port
EXPOSE 7681

# Set data directory to standard Unix location
ENV AXONO_DATA_DIR=/usr/local/share/axono

# Set workspace as default directory for coding tasks
WORKDIR /workspace

# Run axono via ttyd (no auth for testing)
CMD ["ttyd", "-W", "-p", "7681", "python", "-m", "axono.main"]
