# --- Open cmt line bellow if run by cmd: python *.py
import sys  # nopep8
sys.path.append(".")  # nopep8
# ----
import os
import sys
from src.pipeline import Pipeline
from dotenv import load_dotenv

load_dotenv()

PROVIDER = "http://0.0.0.0:7545"
CHAIN_ID = 5777
FL_ABI = "/Users/hienhuynhdang/Documents/UIT/kltn/Smart-Contract-Federated-Learning/contract_export/FEBlockchainLearning.json"
# FL_ABI = "./abi/FEBlockchainLearning.json"
DATASET_PATH = ""
STORAGE_PATH = "./storage"
if __name__ == '__main__':
    index = sys.argv[1]
    CALLER = os.getenv(f'CALLER{index}')
    PRIVATE_KEY = os.getenv(f'PRIVATE_KEY{index}')
    pipeline = Pipeline(PROVIDER, CHAIN_ID, CALLER, PRIVATE_KEY, FL_ABI, DATASET_PATH, STORAGE_PATH)
    pipeline()
