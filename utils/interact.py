from web3 import Web3
import json
import pydash as py_
from eth_account import Account


class SmartContractInteractor(object):
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, provider, chain_id, caller, private_key):
        self.w3 = Web3(Web3.HTTPProvider(provider))
        if self.w3.is_connected():
            print("-" * 50)
            print(f"{self.GREEN}Connection Successful{self.RESET}")
            print("-" * 50)
        else:
            print("Connection Failed")
        self.chain_id = chain_id
        self.caller = caller
        self.private_key = private_key

    def __load_contract(self, _path):
        with open(_path, "r") as f:
            data = json.load(f)
            address = py_.get(data, "address")
            abi = py_.get(data, "abi")
            name = py_.get(data, "name")
        contract = self.w3.eth.contract(address=address, abi=abi)
        return contract, name

    def call(self, contract_path, func, *args):
        # load the contract
        contract, name = self.__load_contract(contract_path)
        # init function
        contract_func = contract.functions[func]
        # call
        response = contract_func(*args).call()
        return response

    def transact(self, contract_path, func, *args, value=0):
        # load the contract
        contract, name = self.__load_contract(contract_path)
        # init function
        contract_func = contract.functions[func]
        # nonce
        nonce = self.w3.eth.get_transaction_count(self.caller)
        # build transaction
        built_txn = contract_func(*args).build_transaction({"from": self.caller, "value": value, "nonce": nonce, "gas": 1000000})
        # sign transaction
        signed_txn = Account.sign_transaction(built_txn, self.private_key)
        # send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # wait transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        transaction_hash = py_.get(tx_receipt, "transactionHash").hex()
        # print log
        nonce = py_.get(tx_receipt, "blockNumber")
        gas_used = py_.get(tx_receipt, "gasUsed")
        from_ = py_.get(tx_receipt, "from")
        to_ = py_.get(tx_receipt, "to")
        status = py_.get(tx_receipt, "status")
        print("-" * 85)
        print(f"Transaction sent: {self.BLUE}{self.BOLD}{transaction_hash}{self.RESET}")
        print(f"\tGas price: {self.BLUE}{self.BOLD}{self.w3.from_wei(self.w3.eth.gas_price, 'gwei')}{self.RESET} gwei   Gas limit: {self.BLUE}{self.BOLD}10000000{self.RESET}   Nonce: {self.BLUE}{self.BOLD}{nonce}{self.RESET}")
        if status:
            print(f"\t{name}.{func} {self.GREEN}{self.BOLD}confirmed{self.RESET}   Block: {self.BLUE}{self.BOLD}{nonce}{self.RESET}   Gas used: {self.BLUE}{self.BOLD}{gas_used}{self.RESET} ({self.BLUE}{self.BOLD}{gas_used/10000000*100}%{self.RESET})")
        else:
            print(f"\t{name}.{func} {self.RED}{self.BOLD}failed{self.RESET}   Block: {self.BLUE}{self.BOLD}{nonce}{self.RESET}   Gas used: {self.BLUE}{self.BOLD}{gas_used}{self.RESET} ({self.BLUE}{self.BOLD}{gas_used/10000000*100}%{self.RESET})")
        print(f"\tFrom: {self.BLUE}{self.BOLD}{from_}{self.RESET}")
        print(f"\tTo: {self.BLUE}{self.BOLD}{to_}{self.RESET}")
        print("-" * 85)
        return tx_receipt

    def transact1(self, contract_path, func, *args, value=None):
        # load the contract
        contract, name = self.__load_contract(contract_path)
        # init function
        contract_func = contract.functions[func]
        # transact
        if value:
            tx_hash = contract_func(*args).transact({"from": self.caller, "value": value})
        else:
            tx_hash = contract_func(*args).transact({"from": self.caller})
        # wait transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        transaction_hash = py_.get(tx_receipt, "transactionHash").hex()
        # print log
        nonce = py_.get(tx_receipt, "blockNumber")
        gas_used = py_.get(tx_receipt, "gasUsed")
        from_ = py_.get(tx_receipt, "from")
        to_ = py_.get(tx_receipt, "to")
        status = py_.get(tx_receipt, "status")
        print("-" * 85)
        print(f"Transaction sent: {self.BLUE}{self.BOLD}{transaction_hash}{self.RESET}")
        print(f"\tGas price: {self.BLUE}{self.BOLD}{self.w3.from_wei(self.w3.eth.gas_price, 'gwei')}{self.RESET} gwei   Gas limit: {self.BLUE}{self.BOLD}10000000{self.RESET}   Nonce: {self.BLUE}{self.BOLD}{nonce}{self.RESET}")
        if status:
            print(f"\t{name}.{func} {self.GREEN}{self.BOLD}confirmed{self.RESET}   Block: {self.BLUE}{self.BOLD}{nonce}{self.RESET}   Gas used: {self.BLUE}{self.BOLD}{gas_used}{self.RESET} ({self.BLUE}{self.BOLD}{gas_used/10000000*100}%{self.RESET})")
        else:
            print(f"\t{name}.{func} {self.RED}{self.BOLD}failed{self.RESET}   Block: {self.BLUE}{self.BOLD}{nonce}{self.RESET}   Gas used: {self.BLUE}{self.BOLD}{gas_used}{self.RESET} ({self.BLUE}{self.BOLD}{gas_used/10000000*100}%{self.RESET})")
        print(f"\tFrom: {self.BLUE}{self.BOLD}{from_}{self.RESET}")
        print(f"\tTo: {self.BLUE}{self.BOLD}{to_}{self.RESET}")
        print("-" * 85)
        return tx_receipt


if __name__ == "__main__":
    contract_path = "/Users/hienhuynhdang/Documents/UIT/kltn/FLClient/abi/Dummy.json"
    provider = "http://0.0.0.0:7545"
    chain_id = 5777
    account_address = "0x52942aeaC70A9848A18BF8d6ed6aCbbf867D0dF9"
    account_private_key = "0xf10bd67ade7cb5c10813adfaedf005832dc3b1759a9412f19d757293c31e31e5"
    smc = SmartContractInteractor(provider, chain_id, account_address, account_private_key)
    bf_number = smc.call(contract_path, "number")
    print(f"Before : {bf_number}")
    tx_receipt = smc.transact(contract_path, "storeNumber", 12)
    af_number = smc.call(contract_path, "number")
    print(f"After : {af_number}")
