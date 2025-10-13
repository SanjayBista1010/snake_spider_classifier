import logging
import os
from datetime import datetime

# -----------------------------
# 1️⃣ Setup log folder
# -----------------------------
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# -----------------------------
# 2️⃣ Log file name
# -----------------------------
log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, log_file)

# -----------------------------
# 3️⃣ Configure logging
# -----------------------------
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',  # append mode
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -----------------------------
# 4️⃣ Example usage
# -----------------------------
logging.info("Logger initialized successfully!")
