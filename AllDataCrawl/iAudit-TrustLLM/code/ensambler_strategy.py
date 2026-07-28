import collections
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import random
import re


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


class Majority:
    def predict(self, results):
        items = collections.defaultdict(list)
        y_real = {}
        for prompt_id, result in results.items():
            for eid in result:
                items[eid].append(result[eid]["pre_label"])
                y_real[eid] = result[eid]["y_real"]

        statistics = {}
        pre_label = {}
        for eid in items:
            voting_top_2 = collections.Counter(items[eid]).most_common(2)
            statistics[eid] = collections.Counter(items[eid])
            if len(voting_top_2) == 2:
                # assert voting_top_2[0][1] == voting_top_2[1][1]
                print(voting_top_2)
                if voting_top_2[0][1] == voting_top_2[1][1]:
                    print(voting_top_2)
                    voting_top_2[0] = ("vulnerable", voting_top_2[0][1])

            pre_label[eid] = voting_top_2[0][0]
            statistics[eid]["pre_label"] = pre_label[eid]
            statistics[eid]["label"] = y_real[eid]
        y_true, y_pred = [], []
        for eid in y_real:
            y_true.append(y_real[eid])
            y_pred.append(pre_label[eid])
        y_true = [1 if x == "vulnerable" else 0 for x in y_true]
        y_pred = [1 if x == "vulnerable" else 0 for x in y_pred]
        return self.performance(y_true, y_pred), statistics

    def evaluate(self, results):
        post_process_res = results.copy()
        for file_name in results:
            items = results[file_name]
            for hash_id in items:
                predictions = items[hash_id]["response_list"]
                flatten_predictions = collections.defaultdict(list)
                count_list = []
                for k, response in enumerate(predictions):
                    flatten_predictions[self.extract_label(response)].append(k)
                    count_list.append(self.extract_label(response))
                voting_top_2 = collections.Counter(count_list).most_common(2)
                if len(voting_top_2) == 2:
                    # assert voting_top_2[0][1] == voting_top_2[1][1]
                    print(voting_top_2)
                    if voting_top_2[0][1] == voting_top_2[1][1]:
                        print(voting_top_2)
                        voting_top_2[0] = ("vulnerable", voting_top_2[0][1])
                pre_dicted = voting_top_2[0][0]
                confidence_score = len(flatten_predictions[pre_dicted]) / (
                    len(flatten_predictions["vulnerable"] + flatten_predictions["safe"])
                )
                # random.seed(1)
                # i = random.choice( flatten_predictions[pre_dicted] )
                post_process_res[file_name][hash_id]["pre_label"] = pre_dicted
                post_process_res[file_name][hash_id][
                    "pre_label_confidence"
                ] = confidence_score
        return post_process_res

    def extract_label(self, text):
        """
        Extracts the label from the given text. The label is expected to be in the format 'The label is [label]',
        where [label] can be either 'safe' or 'vulnerable'.

        Parameters:
        text (str): The text from which to extract the label.

        Returns:
        str: The extracted label ('safe', 'vulnerable', or 'Label not found' if not found).
        """
        # Regular expression pattern to find the label
        text = text.lower()
        pattern = r"the label is (safe|vulnerable)"

        # Search the text for the pattern
        match = re.search(pattern, text)

        # Extract and return the label if found, otherwise return 'Label not found'
        if match:
            return match.group(1)
        else:
            print("Label not found")
            if "is safe" in text:
                return "safe"
            elif "to be safe" in text:
                return "safe"
            else:
                return "vulnerable"

    def performance(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        print(f"F1 Score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"Accuracy: {accuracy}")

        from sklearn.metrics import confusion_matrix

        print(confusion_matrix(y_true, y_pred).ravel())
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return f1, recall, precision, accuracy, tn, fp, fn, tp
