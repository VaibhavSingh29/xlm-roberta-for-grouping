from tqdm import tqdm

with open('groundtruth.txt', 'r', encoding='utf-8') as file:
   sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences]

new_sentences = []
for sentence in sentences:
    words = sentence.split()
    words= words[:-1]
    sent = ' '.join(words)
    new_sentences.append(sent + '\n')

with open('ungrouped.txt', 'w', encoding='utf-8') as file:
    for sent in new_sentences:
        file.write(sent)

with open('grouped.txt', 'r', encoding='utf-8') as file:
   grouped = file.readlines()

grouped = [sentence.strip() for sentence in grouped]

total_list = []
for i in tqdm(range(len(grouped))):
    sentence = grouped[i]
    words = sentence.split()
    words = words[:-1]
    label_list = []
    for word in words:
        ls = word.split('_')
        if len(ls) > 1:
            for item in ls:
                label_list.append('[MERGE]')
        else:
            label_list.append('[COPY]')

    total_list.append(label_list)

with open('labels.txt', 'w', encoding='utf-8') as file:
    for label in total_list:
        file.write(' '.join(label) + '\n')