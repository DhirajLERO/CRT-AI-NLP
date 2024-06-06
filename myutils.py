import json
import os
import _jsonnet
from transformers import tokenization_utils


def graph(data, x_labels, bar_labels, y_label, pdf_name):
    import matplotlib as mpl
    mpl.use('Agg')  # to avoid warning if x-server is not available
    import matplotlib.pyplot as plt
    plt.style.use('scripts/rob.mplstyle')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bar_width = .8 / len(data)
    for rowIdx, row in enumerate(data):
        if len(data) == 2 and rowIdx == 0:  # TODO hardcoded
            x = [x - .5 * bar_width for x in range(len(data[0]))]
        elif len(data) == 2 and rowIdx == 1:
            x = [x + .5 * bar_width for x in range(len(data[0]))]
        elif len(data) == 1:
            x = range(len(data[0]))
        ax.bar(x, row, bar_width, label=bar_labels[rowIdx])

    ax.set_xticks(range(len(data[0])))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel(y_label)
    if len(''.join(bar_labels)) > 0:
        leg = ax.legend()
        leg.get_frame().set_linewidth(1.5)

    fig.savefig(pdf_name, bbox_inches='tight')


def train(name, config, config_done=False, param_config=''):
    path = 'configs/' + name + '.json'
    if config_done:
        json.dump(config, open(path, 'w'), indent=4)
    else:
        json.dump({name: config}, open(path, 'w'), indent=4)
    if param_config != '':
        name = name + '_' + param_config.split('/')[-1].replace('.json', '').replace('/', '_')
    cmd = 'python3 train.py --dataset_config ../2023/' + path + ' --name ' + name
    if param_config != '':
        cmd += ' --parameters_config ' + param_config
    if getModel(name) == '':
        print(cmd)


def getModel(name):
    modelDir = '../machamp/logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.pt'
            if os.path.isfile(modelPath):
                return modelPath
    return ''


def tok(text):
    misc = []
    sent = []
    curMisc = []
    curWord = ''
    prevCat = 'w'
    for char in text:
        if char == ' ':
            if curWord != '':
                sent.append(curWord)
                misc.append(curMisc)
                curWord = ''
                curMisc = []
            curMisc.append(char)
            prevCat = 'w'
        else:
            if prevCat == 'o' or curWord == '':
                curWord += char
            else:
                sent.append(curWord)
                misc.append(curMisc)
                curWord = char
                curMisc = []
            prevCat = 'o'
    if curWord != '':
        sent.append(curWord)
        misc.append(curMisc)
    return sent, misc


def clean_text(text):
    new_text = ''
    for char in text:
        # No clue why we need the second check
        if not tokenization_utils._is_control(char) and not ord(char) <= 31:
            new_text += char
    return new_text.replace('\n', ' ').replace('\t', ' ').strip()


def write(data, path):
    out_file = open(path, 'w')
    for line in data:
        out_file.write(line + '\n')
    out_file.close()


def load_json(path: str):
    """
    Loads a jsonnet file through the json package and returns a dict.

    Parameters
    ----------
    path: str
        the path to the json(net) file to load
    """
    return json.loads(_jsonnet.evaluate_snippet("", '\n'.join(open(path).readlines())))


def makeParams(defaultPath, mlm):
    config = load_json(defaultPath)
    config['transformer_model'] = mlm
    tgt_path = 'configs/params.' + mlm.replace('/', '_') + '.json'
    if not os.path.isfile(tgt_path):
        json.dump(config, open(tgt_path, 'w'), indent=4)
    return '../2023/' + tgt_path


bests = ['task2_1_seq-bio_shared_bert-base-multilingual-cased',
         'task3_1_classification_shared_bert-base-multilingual-cased',
         "task3_2_multi-clas_it_bert-base-multilingual-cased", "task3_2_multi-clas_ru_bert-base-multilingual-cased",
         "task3_2_multi-clas_fr_bert-base-multilingual-cased", "task3_2_multi-clas_ge_bert-base-multilingual-cased",
         "task3_2_multi-clas_en_bert-base-multilingual-cased", "task3_2_multi-clas_po_bert-base-multilingual-cased",
         'task3_3_classification_shared_bert-base-multilingual-cased',
         'task4_1_multiclas_comb_bert-base-multilingual-cased',
         'task5_1_classification_1234567_bert-base-multilingual-cased',
         'task5_2_seq-bio_base_bert-base-multilingual-cased', 'task6_1_seq-bio_mono_bert-base-multilingual-cased',
         'task6_2_seq_NER-TRAIN-JUDGEMENT_bert-base-multilingual-cased',
         'task6_2_seq_NER-TRAIN-PREAMBLE_bert-base-multilingual-cased',
         'task6_3_classification_mono_bert-base-multilingual-cased',
         'task7_1_classification_section_bert-base-multilingual-cased',
         'task7_2_classification_base_bert-base-multilingual-cased',
         'task8_1_seq-bio_base_bert-base-multilingual-cased', 'task8_2_seq-bio_base_bert-base-multilingual-cased',
         'task9_1_regression_shared_bert-base-multilingual-cased',
         'task10_123_classification_multi_bert-base-multilingual-cased',
         'task11_0_regression_0_bert-base-multilingual-cased', 'task11_1_regression_0_bert-base-multilingual-cased',
         'task11_2_classification_0_bert-base-multilingual-cased', 'task11_3_regression_0_bert-base-multilingual-cased',
         "task12_1_classification_am_bert-base-multilingual-cased",
         "task12_1_classification_dz_bert-base-multilingual-cased",
         "task12_1_classification_ha_bert-base-multilingual-cased",
         "task12_1_classification_ig_bert-base-multilingual-cased",
         "task12_1_classification_kr_bert-base-multilingual-cased",
         "task12_1_classification_ma_bert-base-multilingual-cased",
         "task12_1_classification_pcm_bert-base-multilingual-cased",
         "task12_1_classification_pt_bert-base-multilingual-cased",
         "task12_1_classification_sw_bert-base-multilingual-cased",
         "task12_1_classification_twi_bert-base-multilingual-cased",
         "task12_1_classification_yo_bert-base-multilingual-cased"]

selectedLMs = ['bert-base-multilingual-cased', 'microsoft/mdeberta-v3-base', 'studio-ousia/mluke-large',
               'studio-ousia/mluke-large-lite', 'xlm-roberta-large', 'microsoft/infoxlm-large',
               'facebook/mbart-large-50']

multiRegressive = ['Helsinki-NLP/opus-mt-mul-en', 'bigscience/bloom-560m', 'facebook/mbart-large-50',
                   'facebook/mbart-large-50-many-to-many-mmt', 'facebook/mbart-large-50-many-to-one-mmt',
                   'facebook/mbart-large-50-one-to-many-mmt', 'facebook/mbart-large-cc25', 'facebook/mgenre-wiki',
                   'facebook/nllb-200-distilled-600M', 'facebook/xglm-564M', 'facebook/xglm-564M', 'google/byt5-base',
                   'google/byt5-small', 'google/canine-c', 'google/canine-s', 'google/mt5-base', 'google/mt5-small',
                   'sberbank-ai/mGPT']
multiAutoencoder = ['Peltarion/xlm-roberta-longformer-base-4096', 'bert-base-multilingual-cased',
                    'bert-base-multilingual-uncased', 'cardiffnlp/twitter-xlm-roberta-base',
                    'distilbert-base-multilingual-cased', 'google/rembert', 'microsoft/infoxlm-base',
                    'microsoft/infoxlm-large', 'microsoft/mdeberta-v3-base', 'setu4993/LaBSE',
                    'studio-ousia/mluke-base', 'studio-ousia/mluke-base-lite', 'studio-ousia/mluke-large',
                    'studio-ousia/mluke-large-lite', 'xlm-mlm-100-1280', 'xlm-roberta-base', 'xlm-roberta-large']
too_large = ['facebook/xlm-roberta-xxl', 'facebook/xlm-roberta-xl', 'google/byt5-xxl', 'google/mt5-xxl',
             'google/mt5-xl', 'google/byt5-xl', 'google/byt5-large', 'google/mt5-large', 'facebook/nllb-200-1.3B',
             'facebook/nllb-200-3.3B', 'facebook/nllb-200-distilled-1.3B']

