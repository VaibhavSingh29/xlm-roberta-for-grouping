import torch

def merge_sentence(split_sentence, predictions_list):
    merged_sentence = []
    merged_string = ''
    for word, label in zip(split_sentence, predictions_list):
        if label == '[COPY]':
            if merged_string:
                merged_string = merged_string[:-1]
                merged_sentence.append(merged_string)
                merged_string = ''
                merged_sentence.append(word)
            else:
                merged_sentence.append(word)
        elif label == '[MERGE]':
            merged_string = merged_string + word + '_'
    return ' '.join(merged_sentence)


def point_inference(text, model, tokenizer, tags, device):
    '''
    Generate grouping with fine-tuned model for a standalone example
    '''
    tokenized_input = tokenizer(text.split(), return_tensors='pt', is_split_into_words=True)
    word_ids = tokenized_input.word_ids(0)
    outputs = model(tokenized_input.input_ids.to(device)).logits
    predictions = torch.argmax(outputs, dim=-1)[0]
    filtered_predictions = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id == None or word_id == previous_word_id:
            continue
        else:
            filtered_predictions.append(predictions[idx])
        previous_word_id = word_id
    
    print(f"Input Sentence: {text}")
    zipped_list = list(zip(text.split(), [tags[p] for p in filtered_predictions]))
    print(f"Word-wise predictions\n{'-'*20}")
    for item in zipped_list:
        print(item)
    print(f"Grouped Sentence: {merge_sentence(text.split(), [tags[p] for p in filtered_predictions])}")
    return [tags[p] for p in filtered_predictions]
