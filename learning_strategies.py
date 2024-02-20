from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix
import json, evaluate
import numpy as np


def load_dataset(task, language, domain):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['train']['text2'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []
    data['dev']['text2'] = []
    data['test']['label'] = []
    data['test']['text'] = []
    data['test']['text2'] = []

    for l in language:
        for d in domain:
            try:
                with open('data/'+l+'-'+d+'-'+task+'.json') as filehandle:
                    json_data = json.load(filehandle)
            except:
                print('The file '+l+'-'+d+'-'+task+'.json is not available.')
                continue

            print('File ' + l + '-' + d + '-' + task + '.json loaded.')
            for sample in json_data['train']:
                data['train']['text'].append(sample[0])
                data['train']['text2'].append(sample[1])
                data['train']['label'].append(sample[2])

            for sample in json_data['dev']:
                data['dev']['text'].append(sample[0])
                data['dev']['text2'].append(sample[1])
                data['dev']['label'].append(sample[2])

            for sample in json_data['test']:
                data['test']['text'].append(sample[0])
                data['test']['text2'].append(sample[1])
                data['test']['label'].append(sample[2])

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def tokenize_sequence(samples):
    return tknz(samples["text"], samples["text2"], padding="max_length", truncation=True)


def load_model(n_lb):
    tokenizer_hf = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=n_lb, ignore_mismatched_sizes=True)

    return tokenizer_hf, model


def load_local_model(path):
    tokenizer_hf = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained(path)

    return tokenizer_hf, model


def load_argument_model(n_lb, path):
    tokenizer_hf = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=n_lb,
                                                               ignore_mismatched_sizes=True)

    return tokenizer_hf, model


def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


def train_model(mdl, tknz, data):

    training_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=1e-7,
        weight_decay=0.01,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True
    )

    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tknz,
        data_collator=DataCollatorWithPadding(tokenizer=tknz),
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer


if __name__ == "__main__":

    PRETRAIN = False
    CONTINUE = False

    # DEFINE EXPERIMENT TASK, LANGUAGE, AND DOMAIN:
    # ari
    task = 'ari'
    # eng, cn
    language = ['eng']
    # essay, fin
    domain = ['essay']

    num_labels = 3

    # LOAD DATA FOR THE MODE
    dataset = load_dataset(task, language, domain)

    if CONTINUE:
        # LOAD PRE_TRAINED ARGUMENT MINING MODEL
        tknz, mdl = load_argument_model(num_labels, 'models/Domain/Cross/eng-fin-essay-ari/checkpoint-19147')
        # tknz, mdl = load_argument_model(num_labels, 'models/_cont/checkpoint-55300')

        # TOKENIZE THE DATA
        tokenized_data = dataset.map(tokenize_sequence, batched=True)

        # TRAIN THE MODEL
        trainer = train_model(mdl, tknz, tokenized_data)

        # GENERATE PREDICTIONS FOR DEV AND TEST
        dev_predictions = trainer.predict(tokenized_data['dev'])
        dev_predict = np.argmax(dev_predictions.predictions, axis=-1)
        test_predictions = trainer.predict(tokenized_data['test'])
        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        mf1_dev = f1_score(tokenized_data['dev']['label'], dev_predict, average='macro')
        mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='macro')

        print('Macro F1 score in', task, language, domain, 'setup, DEV:', mf1_dev, 'TEST:', mf1_test)

    elif PRETRAIN:

        # LOAD PRE_TRAINED XLM-ROBERTA
        tknz, mdl = load_model(num_labels)

        # TOKENIZE THE DATA
        tokenized_data = dataset.map(tokenize_sequence, batched=True)

        # TRAIN THE MODEL
        trainer = train_model(mdl, tknz, tokenized_data)

        # GENERATE PREDICTIONS FOR DEV AND TEST
        dev_predictions = trainer.predict(tokenized_data['dev'])
        dev_predict = np.argmax(dev_predictions.predictions, axis=-1)
        test_predictions = trainer.predict(tokenized_data['test'])
        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        mf1_dev = f1_score(tokenized_data['dev']['label'], dev_predict, average='macro')
        mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='macro')

        print('Macro F1 score in', task, language, domain, 'setup, DEV:', mf1_dev, 'TEST:', mf1_test)

    else:

        path_cn_fin_ari = 'models/Baselines/cn-fin-ari/checkpoint-3260'
        path_eng_essay_ari = 'models/Baselines/eng-essay-ari/checkpoint-7005/'
        path_eng_fin_ari = 'models/Baselines/eng-fin-ari/checkpoint-58400/'

        path_eng_multidomain_ari = 'models/Domain/Multi/eng-ef-ari/checkpoint-14266'

        path_cn_crossdomain_ari = 'models/Domain/Cross/cn-essay-fin-ari/checkpoint-7172'
        path_eng_crossdomain_fe_ari = 'models/Domain/Cross/eng-fin-essay-ari/checkpoint-19147'
        path_eng_crossdomain_ef_ari = 'models/Domain/Cross/eng-essay-fin-ari/checkpoint-21014'

        path_fin_multilang_ari = 'models/Language/Multi/cneng-fin-ari/checkpoint-14448'

        path_fin_crosslang_ce_ari = 'models/Language/Cross/cn-eng-fin-ari/checkpoint-90692'
        path_fin_crosslang_ec_ari = 'models/Language/Cross/eng-cn-fin-ari/checkpoint-43684'

        path_cneng_ef_ari = 'models/All/MultiLD/cneng-ef-ari/checkpoint-26736'

        path_cross_ld_ari_cnf_ene = 'models/All/CrossLD/cnfin-enessay-ari/checkpoint-18680'
        path_cross_ld_ari_ene_cnf = 'models/All/CrossLD/enessay-cnfin-ari/checkpoint-65200'

        path_cross_ld_3_cf_ee_ef_ari = 'models/All/CrossLD/cnfin-enessay-enfin-ari/checkpoint-48664'
        path_cross_ld_3_ee_cf_ef_ari = 'models/All/CrossLD/enessay-cnfin-enfin-ari/checkpoint-37604'

        tknz, mdl = load_local_model(path_eng_multidomain_ari)

        shuffled_dataset = dataset.shuffle(seed=42)

        tokenized_data = shuffled_dataset.map(tokenize_sequence, batched=True)

        trainer = Trainer(mdl)

        dev_predictions = trainer.predict(tokenized_data['dev'])
        dev_predict = np.argmax(dev_predictions.predictions, axis=-1)
        test_predictions = trainer.predict(tokenized_data['test'])
        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        #print(dev_predict)

        #print(test_predict)

        mf1_dev = f1_score(tokenized_data['dev']['label'], dev_predict, average='macro')
        mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='macro')

        print('Macro F1 score in', task, language, domain, 'setup, DEV:', mf1_dev, 'TEST:', mf1_test)
        print('Confusion matrix',  task, language, domain)
        print(confusion_matrix(tokenized_data['test']['label'], test_predict))
