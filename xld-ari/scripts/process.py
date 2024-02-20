import os
import json
import random
import pandas as pd
import numpy as np

eng_essay_ac = {}
eng_essay_ari = {}
num_ac_labels_ene = {'Claim': 0, 'Premise': 0}
num_ari_labels_ene = {'Support': 0, 'Attack': 0, 'None': 0}
for file in os.listdir("data/essays/EN/"):
    if file.split('.')[-1] == 'ann':
        essay_ann = open('data/essays/EN/' + file, 'r')
        eng_essay_ac[file] = {}
        eng_essay_ari[file] = []
        aux = []
        for ln in essay_ann:
            id = ln.split()[0]
            label = ln.split()[1]
            text = ''
            if len(ln.split('\t')) == 3:
                text = ln.split('\t')[2].strip()

            if label == 'MajorClaim' or label == 'Claim' or label == 'Premise':
                if label == 'MajorClaim' or label == 'Claim':
                    lbc = 1
                    num_ac_labels_ene['Claim'] += 1
                else:
                    lbc = 0
                    num_ac_labels_ene['Premise'] += 1
                eng_essay_ac[file][id] = {'label': lbc, 'text': text}

            elif label == 'supports' or label == 'attacks':
                if label == 'supports':
                    lbr = 1
                    num_ari_labels_ene['Support'] += 1
                else:
                    lbr = 2
                    num_ari_labels_ene['Attack'] += 1
                source = ln.split()[2].split(':')[1]
                text1 = eng_essay_ac[file][source]['text']
                dest = ln.split()[3].split(':')[1]
                text2 = eng_essay_ac[file][dest]['text']
                eng_essay_ari[file].append([text1, text2, lbr])
                aux.append([source, dest])

        for prop1 in eng_essay_ac[file]:
            while True:
                prop2 = random.choice(list(eng_essay_ac[file].keys()))
                if num_ari_labels_ene['None'] == 2000:
                    break
                if [prop1, prop2] not in aux and [prop2, prop1] not in aux and prop1 != prop2:
                    eng_essay_ari[file].append([eng_essay_ac[file][prop1]['text'], eng_essay_ac[file][prop2]['text'], 0])
                    num_ari_labels_ene['None'] += 1
                    break

json_object1 = json.dumps(eng_essay_ac, indent=4)
json_object2 = json.dumps(eng_essay_ari, indent=4)

with open("data/essays/eng-essay-ac.json", "w") as outfile:
    outfile.write(json_object1)

with open("data/essays/eng-essay-ari.json", "w") as outfile:
    outfile.write(json_object2)

cn_essay_ac = {}
num_ac_labels_cne = {'Claim': 0, 'Premise': 0}

for file in os.listdir("data/essays/CN/"):
    essay_file = open('data/essays/CN/' + file, 'r')

    for ln in essay_file:
        line_data = json.loads(ln)
        cn_essay_ac[line_data['file']] = []

        for i in range(len(line_data['labels'])):
            if line_data['labels'][i] == 'Elaboration' or line_data['labels'][i] == 'Evidence':
                lb = 0
                num_ac_labels_cne['Premise'] += 1
            elif line_data['labels'][i] == 'Main Idea' or line_data['labels'][i] == 'Conclusion' or line_data['labels'][i] == 'Thesis':
                lb = 1
                num_ac_labels_cne['Claim'] += 1
            else:
                continue
            cn_essay_ac[line_data['file']].append([' '.join(line_data['sents'][i]), lb])



json_object1 = json.dumps(cn_essay_ac, indent=4)

with open("data/essays/cn-essay-ac.json", "w") as outfile:
    outfile.write(json_object1)

no_relation_list = {}
cat_debate_ari = []
num_ari_labels_cad = {'Support': 0, 'Attack': 0, 'None': 0}
for file in os.listdir("data/debate/CAT/"):
    no_cont = 0
    no_relation_list[file] = []
    # open pandas csv
    df = pd.read_csv('data/debate/CAT/' + file)
    # print(df)
    data_list = df.values.tolist()
    for adu in data_list:
        sample = df.loc[df['ID (Chronological)'] == adu[0]]
        t1 = sample['ADU_CAT'].values[0]
        if no_cont < 177:
            if sample['TYPE (Part + Person)'].values[0] != 'CONC':
                no_relation_list[file].append(t1)
                no_cont += 1
        # 1 type of relation
        if sample['RELATED ID'].to_string(index=False) != 'NaN':
            label = sample['ARGUMENTAL RELATION  TYPE'].to_string(index=False)
            if label == 'RA':
                label = 1
            else:
                label = 2
            rel_ids = str(sample['RELATED ID'].values[0]).split(';')
            for id in rel_ids:
                t2 = df.loc[df['ID (Chronological)'] == int(float(id))]['ADU_CAT'].values[0]
                cat_debate_ari.append([t1, t2, label])
                if label == 1:
                    num_ari_labels_cad['Support'] += 1
                else:
                    num_ari_labels_cad['Attack'] += 1
        # 3 types of relation
        if len(adu) == 13:
            if sample['RELATED ID.1'].to_string(index=False) != 'NaN':
                label = sample['ARGUMENTAL RELATION  TYPE.1'].to_string(index=False)
                if label == 'RA':
                    label = 1
                else:
                    label = 2
                rel_ids = str(sample['RELATED ID.1'].values[0]).split(';')
                for id in rel_ids:
                    t2 = df.loc[df['ID (Chronological)'] == int(float(id))]['ADU_CAT'].values[0]
                    cat_debate_ari.append([t1, t2, label])
                    if label == 1:
                        num_ari_labels_cad['Support'] += 1
                    else:
                        num_ari_labels_cad['Attack'] += 1
            if sample['RELATED ID.2'].to_string(index=False) != 'NaN':
                label = sample['ARGUMENTAL RELATION  TYPE.2'].to_string(index=False)
                if label == 'RA':
                    label = 1
                else:
                    label = 2

                rel_ids = str(sample['RELATED ID.2'].values[0]).split(';')
                for id in rel_ids:
                    t2 = df.loc[df['ID (Chronological)'] == int(float(id))]['ADU_CAT'].values[0]
                    cat_debate_ari.append([t1, t2, label])
                    if label == 1:
                        num_ari_labels_cad['Support'] += 1
                    else:
                        num_ari_labels_cad['Attack'] += 1
        # 2 types of relation
        elif len(adu) == 11:
            if sample['RELATED ID.1'].to_string(index=False) != 'NaN':
                #print(sample['RELATED ID.1'])
                label = sample['ARGUMENTAL RELATION  TYPE.1'].to_string(index=False)
                if label == 'RA':
                    label = 1
                else:
                    label = 2
                rel_ids = str(sample['RELATED ID.1'].values[0]).split(';')
                for id in rel_ids:
                    t2 = df.loc[df['ID (Chronological)'] == int(float(id))]['ADU_CAT'].values[0]
                    cat_debate_ari.append([t1, t2, label])
                    if label == 1:
                        num_ari_labels_cad['Support'] += 1
                    else:
                        num_ari_labels_cad['Attack'] += 1


    # print(file)
    # print(num_ari_labels_cad)
key_list = list(no_relation_list.keys())
for i in range(0, len(key_list)):
    if num_ari_labels_cad['None'] == 5121:
        break
    k1 = key_list[i]
    k2 = key_list[(i+1) % len(key_list)]
    k3 = key_list[(i+2) % len(key_list)]

    for j in range(min(len(no_relation_list[k1]), len(no_relation_list[k2]))):
        if num_ari_labels_cad['None'] == 5121:
            break
        t1 = no_relation_list[k1][j]
        t2 = no_relation_list[k2][j]
        cat_debate_ari.append([t1, t2, 0])
        num_ari_labels_cad['None'] += 1

    for k in range(min(len(no_relation_list[k1]), len(no_relation_list[k3]))):
        if num_ari_labels_cad['None'] == 5121:
            break
        t1 = no_relation_list[k1][k]
        t2 = no_relation_list[k3][k]
        cat_debate_ari.append([t1, t2, 0])
        num_ari_labels_cad['None'] += 1

print(len(cat_debate_ari))
print(num_ari_labels_cad)

json_object = json.dumps(cat_debate_ari, indent=4)

with open("data/debate/cat-debate-ari.json", "w") as outfile:
    outfile.write(json_object)

eng_debate_ari = []
num_ari_labels_end = {'Support': 0, 'Attack': 0, 'None': 0}
dataset_debate_eng = pd.DataFrame(columns=['t1', 't2', 'type'])
for file in os.listdir("data/debate/EN/"):
    debate_file = open('data/debate/EN/' + file, 'r')
    if file.endswith('.json'):
        with open("data/debate/EN/" + file) as json_file:
            arg_map = json.load(json_file)
            for node in arg_map['nodes']:
                t1 = None
                t2 = None
                c1 = None
                c2 = None
                link = None
                type = None
                if node['type'] == 'CA' or node['type'] == 'RA' or node['type'] == 'MA':
                    # Link node detected of type CA, RA or MA
                    link = node['nodeID']
                    type = node['type']

                    # Finding edges from/towards the link node
                    for edge in arg_map['edges']:
                        ck1 = False
                        ck2 = False

                        if link == edge['toID']:
                            c1 = edge['fromID']
                            ck1 = True
                            # print('c1', c1)

                        elif link == edge['fromID']:
                            c2 = edge['toID']
                            ck2 = True
                            # print('c2', c2)

                        # Retrieving the text from the nodes
                        if (t1 == None and ck1 == True) or (t2 == None and ck2 == True):
                            for node2 in arg_map['nodes']:
                                if node2['nodeID'] == c1 and node2['type'] == 'I':
                                    t1 = node2['text']
                                    # print('t1', t1)

                                elif node2['nodeID'] == c2 and node2['type'] == 'I':
                                    t2 = node2['text']
                                    # print('t2', t2)

                    if t1 != None and t2 != None:
                        # print(c1, t1, c2, t2, type)
                        dataset_debate_eng = dataset_debate_eng.append({'t1': t1, 't2': t2, 'type': type}, ignore_index=True)
                        if type == 'RA':
                            eng_debate_ari.append([t1, t2, 1])
                            num_ari_labels_end['Support'] += 1
                        elif type == 'CA':
                            eng_debate_ari.append([t1, t2, 2])
                            num_ari_labels_end['Attack'] += 1


# Generating NO samples

# Dataframe conversion to list structure
text1 = dataset_debate_eng['t1'].tolist()
text2 = dataset_debate_eng['t2'].tolist()
text_tuples = list(zip(text1, text2))
target = dataset_debate_eng['type'].tolist()
full_set = []
for i in range(len(target)):
    full_set.append([text_tuples[i], target[i]])

n_ra = 0
n_ca = 0
n_ma = 0
sent_list = []
rel_list = []

# Counting number of items for each label, saving tuples of each type

for element in full_set:

    rel_list.append(element[0])

    if element[0][0] not in sent_list:
        sent_list.append(element[0][0])
    if element[0][1] not in sent_list:
        sent_list.append(element[0][1])

    if element[1] == 'RA':
        n_ra += 1

    elif element[1] == 'CA':
        n_ca += 1

    elif element[1] == 'MA':
        n_ma += 1

# NO samples 30%
n_no = int(((n_ra+n_ca)/0.7)*0.30)+1
n = 0
used_pointers = []
while n < n_no:
    i = np.random.randint(len(sent_list))
    j = np.random.randint(len(sent_list))

    no_tup = (sent_list[i], sent_list[j])
    if (no_tup not in rel_list) and ((i, j) not in used_pointers) and i != j:
        used_pointers.append((i, j))
        n += 1
        dataset_debate_eng = dataset_debate_eng.append({'t1': no_tup[0], 't2': no_tup[1], 'type': 'NO'}, ignore_index=True)
        eng_debate_ari.append([no_tup[0], no_tup[1], 0])
        num_ari_labels_end['None'] += 1

dataset_debate_eng = dataset_debate_eng.dropna()

json_object2 = json.dumps(eng_debate_ari, indent=4)

with open("data/debate/eng-debate-ari.json", "w") as outfile:
    outfile.write(json_object2)

eng_fin_ac = {}
eng_fin_ari = {}
num_ac_labels_enf = {'Claim': 0, 'Premise': 0}
num_ari_labels_enf = {'Support': 0, 'Attack': 0, 'None': 0}

for file in os.listdir("data/finance/EN/"):
    if file.split('.')[-1] == 'ann':
        essay_ann = open('data/finance/EN/' + file, 'r')
        eng_fin_ac[file] = {}
        eng_fin_ari[file] = []
        aux = []
        for ln in essay_ann:
            id = ln.split()[0]
            label = ln.split()[1]
            text = ''
            if len(ln.split()) > 4:
                text = ' '.join(ln.split()[4:]).strip()

            if 'CLAIM' in label or "PREMISE" in label:
                if 'CLAIM' in label:
                    lbc = 1
                    num_ac_labels_enf['Claim'] += 1
                else:
                    lbc = 0
                    num_ac_labels_enf['Premise'] += 1
                eng_fin_ac[file][id] = {'label': lbc, 'text': text}

            elif label == 'SUPPORT' or label == 'ATTACK':
                if label == 'SUPPORT':
                    lbr = 1
                    num_ari_labels_enf['Support'] += 1
                else:
                    lbr = 2
                    num_ari_labels_enf['Attack'] += 1
                source = ln.split()[2].split(':')[1]
                text1 = eng_fin_ac[file][source]['text']
                dest = ln.split()[3].split(':')[1]
                text2 = eng_fin_ac[file][dest]['text']
                eng_fin_ari[file].append([text1, text2, lbr])
                aux.append([source, dest])

        for prop1 in eng_fin_ac[file]:
            numtry = 0
            while True:
                numtry += 1
                prop2 = random.choice(list(eng_fin_ac[file].keys()))
                if num_ari_labels_enf['None'] == 2000 or numtry == 5:
                    break
                if [prop1, prop2] not in aux and [prop2, prop1] not in aux and prop1 != prop2:
                    eng_fin_ari[file].append([eng_fin_ac[file][prop1]['text'], eng_fin_ac[file][prop2]['text'], 0])
                    num_ari_labels_enf['None'] += 1
                    break


json_object1 = json.dumps(eng_fin_ac, indent=4)
json_object2 = json.dumps(eng_fin_ari, indent=4)

with open("data/finance/eng-fin-ac.json", "w") as outfile:
    outfile.write(json_object1)

with open("data/finance/eng-fin-ari.json", "w") as outfile:
    outfile.write(json_object2)


cn_fin_ari = []
num_ari_labels_cnf = {'Support': 0, 'Attack': 0, 'None': 0}
for file in os.listdir("data/finance/CN/"):
    with open('data/finance/CN/' + file) as filehandle:
        fin_data = json.load(filehandle)
    for item in fin_data:
        if item['label'] == 'reply_攻擊':
            cn_fin_ari.append([item['post'], item['reply'], 2])
            num_ari_labels_cnf['Attack'] += 1
        elif item['label'] == 'reply_支持':
            cn_fin_ari.append([item['post'], item['reply'], 1])
            num_ari_labels_cnf['Support'] += 1
        else:
            cn_fin_ari.append([item['post'], item['reply'], 0])
            num_ari_labels_cnf['None'] += 1

json_object = json.dumps(cn_fin_ari, indent=4)

with open("data/finance/cn-fin-ari.json", "w") as outfile:
    outfile.write(json_object)
