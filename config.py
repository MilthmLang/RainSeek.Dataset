from dotenv import load_dotenv
import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(levelname)s | %(message)s'
)

load_dotenv()

class Config:
    def __init__(self):
        self.qvq_model_path = str(os.environ["QVQ_MODEL_PATH"])
        self.qwen_model_path = str(os.environ["QWEN_MODEL_PATH"])
        self.repo_to_download = str(os.environ["REPO_TO_DOWNLOAD"])
        self.path_to_save = str(os.environ["PATH_TO_SAVE"])

        self.s3_ak = str(os.environ["S3_AK"])
        self.s3_sk = str(os.environ["S3_SK"])
        self.s3_endpoint = str(os.environ["S3_ENDPOINT"])
        self.s3_bucket = str(os.environ["S3_BUCKET"])
        self.s3_list_prefix = str(os.environ["S3_LIST_PREFIX"])

config = Config()
