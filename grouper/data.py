from datasets import DatasetDict, Dataset
import random



with open('ungrouped.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences]

with open('labels.txt', 'r', encoding='utf-8') as file:
    labels = file.readlines()

labels = [label.strip() for label in labels]


sentence_list = [sentence.split() for sentence in sentences]
label_list = [label.split() for label in labels] 

tags = ['[COPY]', '[MERGE]']
index2tag = {idx:tag for idx, tag in enumerate(tags)}
tag2index = {tag:idx for idx, tag in enumerate(tags)}

label_list = [[tag2index[label] for label in inner_list] for inner_list in label_list]

train_ratio = 0.7
val_ratio = 0.2
num_samples = len(sentence_list)
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
zipped_data = list(zip(sentence_list, label_list))
random.shuffle(zipped_data)

train_data = zipped_data[:num_train_samples]
temp_data = zipped_data[num_train_samples:]
val_data = temp_data[:num_val_samples]
test_data = temp_data[num_val_samples:]

train_sentence_list, train_label_list = zip(*train_data)
val_sentence_list, val_label_list = zip(*val_data)
test_sentence_list, test_label_list = zip(*test_data)

data = {
    "train": {'text': list(train_sentence_list), 'labels': list(train_label_list)},
    "validation": {'text': list(val_sentence_list), 'labels': list(val_label_list)},
    "test": {'text': list(test_sentence_list), 'labels': list(test_label_list)}
}

custom_dataset = DatasetDict()
for split in data:
    custom_dataset[split] = Dataset.from_dict(data[split])

custom_dataset.save_to_disk(f"dataset")