import os
import time
import shutil

import pydash as py_
import yaml
from bson.objectid import ObjectId
from utils.interact import SmartContractInteractor
from .image_transform import ImageTransform
from .model_trainer import ModelTrainer
from .dataset import Dataset
from web3 import Web3
import requests
import torch

ROUND_STATUS = {
    "Ready": 0,
    "Training": 1,
    "TrainingFailed": 2,
    "Checking": 3,
    "Checked": 4,
    "Aggregating": 5,
    "Testing": 6,
    "End": 7,
    "ChoseAggregator": 8
}


class Pipeline(object):
    def __init__(self, provider, chain_id, caller, private_key, fl_contract, dataset_path, storage_path):
        self.caller = caller
        self.smc = SmartContractInteractor(provider, chain_id, caller, private_key)
        self.fl_contract = fl_contract
        self.storage_path = f"{storage_path}/{self.caller}"
        self.last_balance = self.smc.get_balance(self.caller)
        self.__init_dataset_the_first_time(dataset_path)
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

    def __upload_file(self, path_):
        url = "http://54.251.217.42:7020/v1/api/fl/upload-file"
        files = {'file': open(path_, 'rb')}
        response = requests.request("POST", url, files=files)
        print(response.json())

    def __download_file(self, file_name):
        url = f"http://54.251.217.42:7020/static/{file_name}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{self.storage_path}/{file_name}", 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {file_name} successfully.")
        else:
            raise Exception("Error downloading file")
        return f"{self.storage_path}/{file_name}"

    def encode_id(self, object_id: str):
        # from str to int
        integer_id = int(str(object_id), 16)
        return integer_id

    def decode_id(self, integer_id: int):
        # from int to str
        object_id = hex(integer_id)[2:]
        return object_id

    def __get_ss_info(self, ss_id):
        ss_info = self.smc.call(self.fl_contract, "sessionById", ss_id)
        sessionId = py_.get(ss_info, 0)
        owner = py_.get(ss_info, 1)
        performanceReward = py_.get(ss_info, 2)
        baseReward = py_.get(ss_info, 3)
        maxRound = py_.get(ss_info, 4)
        currentRound = py_.get(ss_info, 5)
        maxTrainerInOneRound = py_.get(ss_info, 6)
        status = py_.get(ss_info, 7)
        resp = {
            "sessionId": sessionId,
            "owner": owner,
            "performanceReward": performanceReward,
            "baseReward": {
                "trainingReward": baseReward[0],
                "checkingReward": baseReward[1],
                "aggregatingReward": baseReward[2],
                "testingReward": baseReward[3]
            },
            "maxRound": maxRound,
            "currentRound": currentRound,
            "maxTrainerInOneRound": maxTrainerInOneRound,
            "status": status
        }
        return resp

    def __get_current_status(self, ss_id):
        ss_info = self.__get_ss_info(ss_id)
        return py_.get(ss_info, "status")

    def __get_round(self, ss_id):
        ss_info = self.__get_ss_info(ss_id)
        return py_.get(ss_info, "currentRound")

    def __waiting_stage(self, ss_id, expect_stage):
        print(f"Waiting the stage {list(ROUND_STATUS.keys())[expect_stage]}")
        while True:
            time.sleep(3)
            stage = self.__get_current_status(ss_id)
            if stage == expect_stage:
                break
        print(f"The stage turned into {list(ROUND_STATUS.keys())[expect_stage]}")
        return

    def __init_dataset_the_first_time(self, dataset_path):
        image_transform = ImageTransform()
        self.num_to_label, self.label_to_num = self.__init_label(dataset_path)
        self.train_dataset = Dataset(image_transform, self.label_to_num, Dataset.TRAIN, dataset_path)
        self.val_dataset = Dataset(image_transform, self.label_to_num, Dataset.VAL, dataset_path)
        self.test_dataset = Dataset(image_transform, self.label_to_num, Dataset.TEST, dataset_path)

    def __init_label(self, dataset_path):
        with open(f"{dataset_path}/data.yaml", 'r') as file:
            data = yaml.safe_load(file)
        num_to_label = {i: name for i, name in enumerate(data['names'])}
        label_to_num = {name: i for i, name in enumerate(data['names'])}
        return num_to_label, label_to_num

    def __call__(self):
        ss_id = self.__choose_session()
        # 1: [SMC] applySession() -> stage = Ready -> waiting Training
        self.__apply_session(ss_id)
        self.__waiting_stage(ss_id, ROUND_STATUS["Training"])
        time.sleep(7)
        # LOOP
        while True:
            current_round = self.__get_round(ss_id) + 1
            print(f"ROUND {current_round}")
            # 2: [SMC] getDataDoTraining() -> get global model + params
            self.global_path, param_path = self.__download_global_model(ss_id)
            # 3: [Local] local training
            new_params_path = self.__train_model(self.global_path, param_path, round=current_round)
            # 4: [BE] Upload Param to server
            new_param_id = self.__upload_params(new_params_path)
            # 5: [SMC] submitUpdate() -> submit param_id -> waiting stage Checking
            self.__submit_update(ss_id, new_param_id)
            self.__waiting_stage(ss_id, ROUND_STATUS["Checking"])
            time.sleep(7)
            # 6: [SMC] getDataDoChecking() -> return at least 4 params
            print(f"ss_id: {ss_id}, type: {type(ss_id)}")
            print(f"caller: {self.caller}, type: {type(self.caller)}")
            checking_ids = self.__get_ids_checking(ss_id)
            # 7: [Local] local check
            bad_trainers = self.__local_check(checking_ids)
            # 8: [SMC] submitBadTrainer() -> waiting Checked
            self.__submit_bad_trainer(ss_id, bad_trainers)
            self.__waiting_stage(ss_id, ROUND_STATUS["ChoseAggregator"])
            time.sleep(7)
            # 9: [SMC] check if can Aggregate()
            if self.__check_if_can_aggregate(ss_id) == True:
                # 10: [SMC] getDataDoAggregate()
                params_id = self.__get_data_for_aggregating(ss_id)
                # 11 [Local] FedAvg
                new_global_model_path, bad_update_id = self.__fed_avg(ss_id, params_id)
                # 12 [BE] upload global params
                new_global_model_param_id = self.__upload_global_model_params(new_global_model_path)
                # 13: [SMC] submitAggregate()
                self.__submit_new_global_model(ss_id, new_global_model_param_id, bad_update_id)
            # 14: waiting Testing
            self.__waiting_stage(ss_id, ROUND_STATUS["Testing"])
            time.sleep(7)
            # 15: [Local] Testing
            global_param_id, params_id = self.__get_data_testing(ss_id)
            test_result = self.__local_testing(ss_id, global_param_id, params_id)
            # 16: [SMC] submitScore()
            self.__submit_scores(ss_id, test_result)
            time.sleep(7)
            # 17: [SMC] claimReward()
            self.__claim_performance_reward(ss_id, current_round - 1)
            ###
            if current_round == self.__get_ss_info(ss_id)["maxRound"]:
                self.__waiting_stage(ss_id, ROUND_STATUS["End"])
                time.sleep(7)
                break
            else:
                self.__waiting_stage(ss_id, ROUND_STATUS["Training"])
                time.sleep(7)

        # END LOOP
        return

    def __choose_session(self):
        # get all session
        all_sessions = self.smc.call(self.fl_contract, "allSession")
        all_ss_id = {}
        print("Existing session:")
        for index, session in enumerate(all_sessions):
            ss_id = py_.get(session, 0)
            all_ss_id[str(index)] = ss_id
            print(f"\t{index}. {self.decode_id(ss_id)}")
        # index = input("Enter index: ")
        index = "0"
        ss_id = all_ss_id[index]
        print(f"Chose session {self.decode_id(ss_id)} -> Applying...")
        return py_.to_integer(ss_id)

    def __apply_session(self, ss_id):
        # get stake value
        ss_info = self.__get_ss_info(ss_id)
        maxTrainerInOneRound = py_.get(ss_info, "maxTrainerInOneRound")
        trainingReward = py_.get(ss_info, "baseReward.trainingReward")
        stake_value = maxTrainerInOneRound * trainingReward
        print(f"ss_id: {ss_id}, type {type(ss_id)}")
        print("stake value", stake_value, type(stake_value))
        # apply session
        tx_receipt = self.smc.transact(self.fl_contract, "applySession", ss_id, value=stake_value)
        if not bool(int(tx_receipt['status'])):
            print("-" * 50, f"\nApply Session {ss_id} Failed\n", "-" * 50)
            raise Exception("Apply Session Failed")
        print("-" * 50, f"\nApply Session {ss_id} Successfully\n", "-" * 50)
        return bool(int(tx_receipt['status']))

    def __download_global_model(self, ss_id):
        global_id, param_id = self.smc.call(self.fl_contract, "getDataDoTraining", ss_id)
        print(f"Downloading global model {self.decode_id(global_id)}")
        global_path = self.__download_file(f"{self.decode_id(global_id)}.pt")
        print(f"Downloading param id {self.decode_id(param_id)}")
        param_path = self.__download_file(f"{self.decode_id(param_id)}.pth")
        return global_path, param_path

    def __train_model(self, global_path, param_path, round=1):
        print(f"Start training model round {round}")
        # TODO: choose learning rate
        learning_rate = 0.001
        # TODO: choose num_epoch
        num_epoch = 3
        batch_size = 5
        # init train obj
        model_trainer = ModelTrainer(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epoch=num_epoch,
            pretrained_path=param_path,
            network_path=global_path,
            output_path=self.storage_path)
        new_param_path = model_trainer.train(include_test=False)
        print(f"End training model round {round}")
        return new_param_path

    def __upload_params(self, params_path):
        # TODO: upload params to server
        self.__upload_file(params_path)
        param_id = os.path.basename(params_path).split(".pth")[0]
        new_param_id = self.encode_id(param_id)
        return new_param_id

    def __submit_update(self, ss_id, param_id):
        # get stake value
        ss_info = self.__get_ss_info(ss_id)
        checking_reward = py_.get(ss_info, "baseReward.checkingReward")
        stake_value = checking_reward * 2
        print(f"ss_id: {ss_id}, type {type(ss_id)}")
        print("stake value", stake_value, type(stake_value))
        # submit update
        tx_receipt = self.smc.transact(self.fl_contract, "submitUpdate", ss_id, param_id, value=stake_value)
        if not bool(int(tx_receipt['status'])):
            print("-" * 50, f"\nSubmit Update {ss_id} Failed\n", "-" * 50)
            raise Exception("Submit Update Failed")
        print("-" * 50, f"\nSubmit Update {ss_id} Successfully\n", "-" * 50)
        return bool(int(tx_receipt['status']))

    def __get_ids_checking(self, ss_id):
        print("Getting params id for checking...")
        caller = Web3.to_checksum_address(self.caller)
        params_id = self.smc.call(self.fl_contract, "getDataDoChecking", ss_id, caller)
        print(f"Got params id for checking: {params_id}")
        return params_id

    def __local_check(self, checking_ids):
        bad_trainers = []
        print("Start checking")
        for checking_id in checking_ids:
            checking_id = self.decode_id(checking_id)
            print(f"Downloading param of checking id: {checking_id}...")
            path_file = self.__download_file(f"{checking_id}.pth")
            # TODO: local check
            print("Checking...")
            result_check = True
            print(f"Is valid: {result_check}")
            bad_trainers.append(result_check)
        print("End checking")
        return bad_trainers

    def __submit_bad_trainer(self, ss_id, bad_trainers):
        # get stake value
        ss_info = self.__get_ss_info(ss_id)
        testing_reward = py_.get(ss_info, "baseReward.testingReward")
        stake_value = testing_reward * 5
        print(f"ss_id: {ss_id}, type {type(ss_id)}")
        print("stake value", stake_value, type(stake_value))
        tx_receipt = self.smc.transact(self.fl_contract, "submitCheckingResult", ss_id, bad_trainers, value=stake_value)
        if not bool(int(tx_receipt['status'])):
            print("-" * 50, f"\nSubmit Checking Result {ss_id} Failed\n", "-" * 50)
            raise Exception("Submit Checking Result Failed")
        print("-" * 50, f"\nSubmit Checking Result {ss_id} Successfully\n", "-" * 50)
        return bool(int(tx_receipt['status']))

    def __check_if_can_aggregate(self, ss_id):
        print("Checking if can aggregate...")
        caller = Web3.to_checksum_address(self.caller)
        result = self.smc.call(self.fl_contract, "checkOpportunityAggregate", ss_id, caller)
        return result

    def __get_data_for_aggregating(self, ss_id):
        print("Getting data for aggregate...")
        params_id = self.smc.call(self.fl_contract, "getDataDoAggregate", ss_id)
        return params_id

    def __fed_avg(self, ss_id, params_id):
        # TODO: Download
        paths = []
        for param_id in params_id:
            param_id = self.decode_id(param_id)
            path_ = self.__download_file(f"{param_id}.pth")
            paths.append(path_)
        print("Aggregating global model...")
        new_global_model_path = self.__aggregate_model_updates(paths)
        # new_global_model_path = ""
        bad_update_id = []
        return new_global_model_path, bad_update_id

    def __upload_global_model_params(self, new_global_model_path):
        self.__upload_file(new_global_model_path)
        new_global_model_param_id = os.path.basename(new_global_model_path).split(".pth")[0]
        new_global_model_param_id_encoded = self.encode_id(new_global_model_param_id)
        return new_global_model_param_id_encoded

    def __submit_new_global_model(self, ss_id, new_global_model_param_id, bad_update_id):
        # submit update
        tx_receipt = self.smc.transact(self.fl_contract, "submitAggregate", ss_id, new_global_model_param_id, bad_update_id)
        if not bool(int(tx_receipt['status'])):
            print("-" * 50, f"\nSubmit Aggregate {ss_id} Failed\n", "-" * 50)
            raise Exception("Submit Aggregate Failed")
        print("-" * 50, f"\nSubmit Aggregate {ss_id} Successfully\n", "-" * 50)
        return bool(int(tx_receipt['status']))

    def __get_data_testing(self, ss_id):
        print("Getting params id for testing...")
        caller = Web3.to_checksum_address(self.caller)
        global_param_id, params_id = self.smc.call(self.fl_contract, "getDataDoTesting", ss_id, caller)
        return global_param_id, params_id

    def __local_testing(self, ss_id, global_param_id, params_id):
        print("Start testing...")
        learning_rate = 0.001
        num_epoch = 3
        batch_size = 5
        # test global model
        print(f"Downloading param id {self.decode_id(global_param_id)}")
        global_param_path = self.__download_file(f"{self.decode_id(global_param_id)}.pth")
        model_trainer = ModelTrainer(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epoch=num_epoch,
            pretrained_path=global_param_path,
            network_path=self.global_path,
            output_path=self.storage_path)
        result = model_trainer.test()
        # test deletion method
        test_result = []
        for param_id in params_id:
            param_path = self.__download_file(f"{self.decode_id(param_id)}.pth")
            model_trainer = ModelTrainer(
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epoch=num_epoch,
                pretrained_path=param_path,
                network_path=self.global_path,
                output_path=self.storage_path)
            result = model_trainer.test()
            test_result.append(result)
        return test_result

    def __submit_scores(self, ss_id, test_result):
        # submit scores
        tx_receipt = self.smc.transact(self.fl_contract, "submitScores", ss_id, test_result)
        if not bool(int(tx_receipt['status'])):
            print("-" * 50, f"\nSubmit Scores {ss_id} Failed\n", "-" * 50)
            raise Exception("Submit Scores Failed")
        print("-" * 50, f"\nSubmit Scores {ss_id} Successfully\n", "-" * 50)
        return bool(int(tx_receipt['status']))

    def __aggregate_model_updates(self, model_files):
        aggregated_model = None

        for file_path in model_files:
            # Load the model from each file
            model = torch.load(file_path)

            if aggregated_model is None:
                aggregated_model = model
            else:
                # Accumulate the model parameters
                for param_name, param in model.items():
                    aggregated_model[param_name] += param

        # Take the average by dividing by the number of models
        num_models = len(model_files)
        for param_name in aggregated_model:
            aggregated_model[param_name] /= num_models
        save_path = f"{self.storage_path}/{ObjectId()}.pth"
        torch.save(aggregated_model, save_path)
        return save_path

    def __claim_performance_reward(self, ss_id, round):
        # submit update
        tx_receipt = self.smc.transact(self.fl_contract, "claimPerformanceReward", ss_id, round)
        print("Claimming Reward....")
        time.sleep(7)
        balance_in_wei = self.smc.get_balance(self.caller)
        green = '\033[92m'
        red = '\033[91m'
        difference = balance_in_wei - self.last_balance
        color = green if difference > 0 else red
        print("-" * 50, f"\nBalance: \033[92m{balance_in_wei}\033[0m Wei || difference: {color}{difference}\033[0m \n", "-" * 50)
        self.last_balance = balance_in_wei
        return bool(int(tx_receipt['status']))
    ################################################################
