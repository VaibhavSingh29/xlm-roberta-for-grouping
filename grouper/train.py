from transformers import AutoConfig, AutoTokenizer
import torch
from model import Grouper
from inference_utils import point_inference
from datasets import load_from_disk
import numpy as np
from transformers import TrainingArguments
from seqeval.metrics import f1_score
from transformers import DataCollatorForTokenClassification
from transformers import Trainer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tags = ['[COPY]', '[MERGE]']
    index2tag = {idx:tag for idx, tag in enumerate(tags)}
    tag2index = {tag:idx for idx, tag in enumerate(tags)}
    custom_config = AutoConfig.from_pretrained('xlm-roberta-base', num_labels=2, id2label=index2tag, label2id=tag2index)
    model = Grouper.from_pretrained('xlm-roberta-base', config=custom_config)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    def tokenize_and_align(example):
        tokenized_inputs = tokenizer(example['text'], is_split_into_words=True)
        labels = []
        for idx, label in enumerate(example['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(label[word_idx])
                    except:
                        continue
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    dataset = load_from_disk('dataset')

    tokenized_dataset = dataset.map(tokenize_and_align, batched=True, remove_columns='text')

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(index2tag[preds[batch_idx][seq_idx]])
            
            labels_list.append(example_labels)
            preds_list.append(example_preds)
        
        return preds_list, labels_list
    
    def compute_metrics(eval):
        y_pred, y_true = align_predictions(eval.predictions, eval.label_ids)
        return {"f1": f1_score(y_true, y_pred)}

    training_args = TrainingArguments(
        output_dir='finetuned',
        log_level='error',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        logging_steps=len(tokenized_dataset['train'])//32
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer
    )

    trainer.train()

    # point_inference(
    #     'संग्रहालय सोमवार को बंद रहता है',
    #     model,
    #     tokenizer,
    #     tags,
    #     device
    # )
    results = trainer.predict(tokenized_dataset['test'])
    print(results.metrics)

if __name__ == "__main__":
    main()