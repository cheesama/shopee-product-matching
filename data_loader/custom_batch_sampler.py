from torch.utils.data.sampler import BatchSampler

import random


class PositivePairAugBatchSampler(BatchSampler):
    def __init__(self, dataset_df, min_positive_instances=8, num_labels_per_batch=16):
        self.max_iter = len(dataset_df)
        self.min_positive_instances = min_positive_instances
        self.num_labels_per_batch = num_labels_per_batch

        self.label_index_dict = {}  # key: batch, value: [batch_indices]
        for label in dataset_df["label_group"]:
            self.label_index_dict[label] = [index for index in list(dataset_df[dataset_df["label_group"] == label].index) if index < len(dataset_df)]

    def __len__(self):
        return self.max_iter

    def __iter__(self):
        for _ in range(self.max_iter):
            batch_indices = []

            selected_labels = random.choices(
                list(self.label_index_dict.keys()), k=self.num_labels_per_batch
            )

            for label in selected_labels:
                batch_indices.extend(
                    random.choices(
                        self.label_index_dict[label], k=self.min_positive_instances
                    )
                )

            yield batch_indices
