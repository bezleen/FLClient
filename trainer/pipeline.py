import time

import pydash as py_
import yaml

from utils.interact import SmartContractInteractor
from .image_transform import ImageTransform
from .model_trainer import ModelTrainer
from .dataset import Dataset

ROUND_STATUS = {
    "Ready": "Ready",
    "Training": "Training",
    "TrainingFailed": "TrainingFailed",
    "Checking": "Checking",
    "Checked": "Checked",
    "Aggregating": "Aggregating",
    "Testing": "Testing",
    "End": "End",
}


class pipeline(object):
    def __init__(self, provider, chain_id, caller, private_key, fl_contract, dataset_path, storage_path):
        self.smc = SmartContractInteractor(provider, chain_id, caller, private_key)
        self.fl_contract = fl_contract
        self.storage_path = storage_path
        self.__init_dataset_the_first_time(dataset_path)
        pass

    def __call__(self, ss_id):
        # 1: [SMC] applySession() -> stage = Ready -> waiting Training
        is_success = self.__apply_session(ss_id)
        if not is_success:
            print("-" * 50, f"\nApply Session {ss_id} Failed\n", "-" * 50)
        print("-" * 50, f"\nApply Session {ss_id} Successfully\n", "-" * 50)
        self.__waiting_stage(ss_id, ROUND_STATUS["Training"])
        # LOOP
        # 2: [SMC] getDataDoTraining() -> get global model + params
        global_path, param_path = self.__download_global_model(ss_id)
        # 3: [Local] local training
        new_params_path = self.__train_model(global_path, param_path)
        # 4: [BE] Upload Param to server
        new_param_id = self.__upload_params(new_params_path)
        # 5: [SMC] submitUpdate() -> submit param_id -> waiting stage Checking
        self.__submit_update(ss_id, new_param_id)
        self.__waiting_stage(ss_id, ROUND_STATUS["Checking"])
        # 6: [SMC] getDataDoChecking() -> return at least 4 params

        # 7: [Local] local check
        # 8: [SMC] submitBadTrainer() -> waiting Aggregating
        # 9: [SMC] applyAggregator()
        # 10: [SMC] if is aggregator -> getDataForAggregating()
        # 11 [Local] FedAvg
        # 12 [BE] upload global params
        # 13: [SMC] submitAggregate() -> waiting Testing
        # 14: [SMC] getDataForTesting()
        # 15: [Local] Testing
        # 16: [SMC] submitScore()
        # 17: [SMC] claimReward()
        # END LOOP

    def __submit_update(self, ss_id, param_id):
        # TODO: get stake value
        stake_value = 0
        # submit update
        try:
            self.smc.transact(self.fl_contract, "submitUpdate", ss_id, value=stake_value)
        except:
            return False
        return True

    def __upload_params(self, params_path):
        # TODO: upload params to server
        new_param_id = ""
        return new_param_id

    def __init_dataset_the_first_time(self, dataset_path):
        image_transform = ImageTransform()
        self.num_to_label, self.label_to_num = self.__init_label(dataset_path)
        self.train_dataset = Dataset(image_transform, self.label_to_num, Dataset.TRAIN, dataset_path)
        self.val_dataset = Dataset(image_transform, self.label_to_num, Dataset.VAL, dataset_path)
        self.test_dataset = Dataset(image_transform, self.label_to_num, Dataset.TEST, dataset_path)

    def __init_label(dataset_path):
        with open(f"{dataset_path}/data.yaml", 'r') as file:
            data = yaml.safe_load(file)
        num_to_label = {i: name for i, name in enumerate(data['names'])}
        label_to_num = {name: i for i, name in enumerate(data['names'])}
        return num_to_label, label_to_num

    def __train_model(self, global_path, param_path):
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
        return new_param_path

    def __apply_session(self, ss_id):
        # TODO: get stake value
        stake_value = self.smc.call(self.fl_contract, "stakeValueById", ss_id)
        # apply session
        try:
            self.smc.transact(self.fl_contract, "applySession", ss_id, value=stake_value)
        except:
            return False
        return True

    def __download_global_model(self, ss_id):
        global_id, param_id = self.smc.call(self.fl_contract, "getDataDoTraining", ss_id)
        # TODO: download model
        global_path = ""
        param_path = ""
        return global_path, param_path

    def __waiting_stage(self, ss_id, expect_stage):
        while True:
            time.sleep(3)
            stage = self.__get_current_status(ss_id)
            if stage == expect_stage:
                break
        return

    def __get_current_status(self, ss_id):
        ss_info = self.__get_ss_info(ss_id)
        return py_.get(ss_info, "status")

    def __get_ss_info(self, ss_id):
        ss_info = self.smc.call(self.fl_contract, "sessionById", ss_id)
        sessionId = py_.get(ss_info, "sessionId")
        owner = py_.get(ss_info, "owner")
        performanceReward = py_.get(ss_info, "performanceReward")
        baseReward = py_.get(ss_info, "baseReward")
        maxRound = py_.get(ss_info, "maxRound")
        currentRound = py_.get(ss_info, "currentRound")
        maxTrainerInOneRound = py_.get(ss_info, "maxTrainerInOneRound")
        status = py_.get(ss_info, "status")
        resp = {
            "sessionId": sessionId,
            "owner": owner,
            "performanceReward": performanceReward,
            "baseReward": baseReward,
            "maxRound": maxRound,
            "currentRound": currentRound,
            "maxTrainerInOneRound": maxTrainerInOneRound,
            "status": status
        }
        return resp
