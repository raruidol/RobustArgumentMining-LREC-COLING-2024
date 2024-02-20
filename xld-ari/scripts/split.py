import os
import json
import random

eng_fin_ari_data = {0: [], 1: [], 2: []}
eng_fin_ari_dataset = {'train': [], 'dev': [], 'test': []}

cn_essay_ac_data = {0: [], 1: []}
cn_essay_ac_dataset = {'train': [], 'dev': [], 'test': []}

eng_essay_ari_data = {0: [], 1: [], 2: []}
eng_essay_ari_dataset = {'train': [], 'dev': [], 'test': []}

cat_debate_ari_data = {0: [], 1: [], 2: []}
cat_debate_ari_dataset = {'train': [], 'dev': [], 'test': []}

eng_debate_ari_data = {0: [], 1: [], 2: []}
eng_debate_ari_dataset = {'train': [], 'dev': [], 'test': []}

eng_fin_ac_data = {0: [], 1: []}
eng_fin_ac_dataset = {'train': [], 'dev': [], 'test': []}

eng_essay_ac_data = {0: [], 1: []}
eng_essay_ac_dataset = {'train': [], 'dev': [], 'test': []}

cn_fin_ari_data = {0: [], 1: [], 2: []}
cn_fin_ari_dataset = {'train': [], 'dev': [], 'test': []}

for file in os.listdir("data/complete/"):
    with open('data/complete/' + file) as filehandle:
        json_data = json.load(filehandle)
    
    if file == 'eng-fin-ari.json':
        for filename in json_data:
            for argpair in json_data[filename]:
                eng_fin_ari_data[argpair[2]].append(argpair)

        random.shuffle(eng_fin_ari_data[0])
        random.shuffle(eng_fin_ari_data[1])
        random.shuffle(eng_fin_ari_data[2])

        eng_fin_ari_dataset['train'] = eng_fin_ari_data[0][0:1600] + eng_fin_ari_data[1][0:3859] + eng_fin_ari_data[2][0:62]
        eng_fin_ari_dataset['dev'] = eng_fin_ari_data[0][1600:1800] + eng_fin_ari_data[1][3859:4341] + eng_fin_ari_data[2][62:70]
        eng_fin_ari_dataset['test'] = eng_fin_ari_data[0][1800:2000] + eng_fin_ari_data[1][4341:4823] + eng_fin_ari_data[2][70:78]

        json_object = json.dumps(eng_fin_ari_dataset, indent=4)

        with open("data/splits/eng-fin-ari.json", "w") as outfile:
            outfile.write(json_object)

    elif file == 'cn-essay-ac.json':
        for filename in json_data:
            for arg in json_data[filename]:
                cn_essay_ac_data[arg[1]].append(arg)

        random.shuffle(cn_essay_ac_data[0])
        random.shuffle(cn_essay_ac_data[1])

        cn_essay_ac_dataset['train'] = cn_essay_ac_data[0][0:16134] + cn_essay_ac_data[1][0:7574]
        cn_essay_ac_dataset['dev'] = cn_essay_ac_data[0][16134:18151] + cn_essay_ac_data[1][7574:8521]
        cn_essay_ac_dataset['test'] = cn_essay_ac_data[0][18151:20168] + cn_essay_ac_data[1][8521:9468]

        json_object = json.dumps(cn_essay_ac_dataset, indent=4)

        with open("data/splits/cn-essay-ac.json", "w") as outfile:
            outfile.write(json_object)

    elif file == 'eng-essay-ari.json':
        for filename in json_data:
            for argpair in json_data[filename]:
                eng_essay_ari_data[argpair[2]].append(argpair)

        random.shuffle(eng_essay_ari_data[0])
        random.shuffle(eng_essay_ari_data[1])
        random.shuffle(eng_essay_ari_data[2])

        eng_essay_ari_dataset['train'] = eng_essay_ari_data[0][0:1600] + eng_essay_ari_data[1][0:2891] + eng_essay_ari_data[2][0:175]
        eng_essay_ari_dataset['dev'] = eng_essay_ari_data[0][1600:1800] + eng_essay_ari_data[1][2891:3252] + eng_essay_ari_data[2][175:197]
        eng_essay_ari_dataset['test'] = eng_essay_ari_data[0][1800:2000] + eng_essay_ari_data[1][3252:3613] + eng_essay_ari_data[2][197:219]

        json_object = json.dumps(eng_essay_ari_dataset, indent=4)

        with open("data/splits/eng-essay-ari.json", "w") as outfile:
            outfile.write(json_object)

    if file == 'cat-debate-ari.json':
        for argpair in json_data:
            cat_debate_ari_data[argpair[2]].append(argpair)

        random.shuffle(cat_debate_ari_data[0])
        random.shuffle(cat_debate_ari_data[1])
        random.shuffle(cat_debate_ari_data[2])

        cat_debate_ari_dataset['train'] = cat_debate_ari_data[0][0:4097] + cat_debate_ari_data[1][0:7566] + cat_debate_ari_data[2][0:1553]
        cat_debate_ari_dataset['dev'] = cat_debate_ari_data[0][4097:4609] + cat_debate_ari_data[1][7566:8512] + cat_debate_ari_data[2][1553:1747]
        cat_debate_ari_dataset['test'] = cat_debate_ari_data[0][4609:5121] + cat_debate_ari_data[1][8512:9458] + cat_debate_ari_data[2][1747:1941]

        json_object = json.dumps(cat_debate_ari_dataset, indent=4)

        with open("data/splits/cat-debate-ari.json", "w") as outfile:
            outfile.write(json_object)

    if file == 'eng-debate-ari.json':
        for argpair in json_data:
            eng_debate_ari_data[argpair[2]].append(argpair)

        random.shuffle(eng_debate_ari_data[0])
        random.shuffle(eng_debate_ari_data[1])
        random.shuffle(eng_debate_ari_data[2])

        eng_debate_ari_dataset['train'] = eng_debate_ari_data[0][0:3514] + eng_debate_ari_data[1][0:6520] + eng_debate_ari_data[2][0:1676]
        eng_debate_ari_dataset['dev'] = eng_debate_ari_data[0][3514:3953] + eng_debate_ari_data[1][6520:7335] + eng_debate_ari_data[2][1676:1886]
        eng_debate_ari_dataset['test'] = eng_debate_ari_data[0][3953:4392] + eng_debate_ari_data[1][7335:8150] + eng_debate_ari_data[2][1886:2096]

        json_object = json.dumps(eng_debate_ari_dataset, indent=4)

        with open("data/splits/eng-debate-ari.json", "w") as outfile:
            outfile.write(json_object)

    elif file == 'eng-fin-ac.json':
        for filename in json_data:
            for id in json_data[filename]:
                arg = json_data[filename][id]
                eng_fin_ac_data[int(arg['label'])].append([arg['text'], arg['label']])

        random.shuffle(eng_fin_ac_data[0])
        random.shuffle(eng_fin_ac_data[1])

        eng_fin_ac_dataset['train'] = eng_fin_ac_data[0][0:4062] + eng_fin_ac_data[1][0:3691]
        eng_fin_ac_dataset['dev'] = eng_fin_ac_data[0][4062:4570] + eng_fin_ac_data[1][3691:4152]
        eng_fin_ac_dataset['test'] = eng_fin_ac_data[0][4570:5078] + eng_fin_ac_data[1][4152:4613]

        json_object = json.dumps(eng_fin_ac_dataset, indent=4)

        with open("data/splits/eng-fin-ac.json", "w") as outfile:
            outfile.write(json_object)

    elif file == 'eng-essay-ac.json':
        for filename in json_data:
            for id in json_data[filename]:
                arg = json_data[filename][id]
                eng_essay_ac_data[int(arg['label'])].append([arg['text'], arg['label']])

        random.shuffle(eng_essay_ac_data[0])
        random.shuffle(eng_essay_ac_data[1])

        eng_essay_ac_dataset['train'] = eng_essay_ac_data[0][0:3066] + eng_essay_ac_data[1][0:1805]
        eng_essay_ac_dataset['dev'] = eng_essay_ac_data[0][3066:3449] + eng_essay_ac_data[1][1805:2031]
        eng_essay_ac_dataset['test'] = eng_essay_ac_data[0][3449:3832] + eng_essay_ac_data[1][2031:2257]

        json_object = json.dumps(eng_essay_ac_dataset, indent=4)

        with open("data/splits/eng-essay-ac.json", "w") as outfile:
            outfile.write(json_object)

    elif file == 'cn-fin-ari.json':
        for argpair in json_data:
            cn_fin_ari_data[argpair[2]].append(argpair)

        random.shuffle(cn_fin_ari_data[0])
        random.shuffle(cn_fin_ari_data[1])
        random.shuffle(cn_fin_ari_data[2])

        cn_fin_ari_dataset['train'] = cn_fin_ari_data[0][0:684] + cn_fin_ari_data[1][0:3676] + cn_fin_ari_data[2][0:2158]
        cn_fin_ari_dataset['dev'] = cn_fin_ari_data[0][684:769] + cn_fin_ari_data[1][3676:4136] + cn_fin_ari_data[2][2158:2428]
        cn_fin_ari_dataset['test'] = cn_fin_ari_data[0][769:854] + cn_fin_ari_data[1][4136:4596] + cn_fin_ari_data[2][2428:2698]

        json_object = json.dumps(cn_fin_ari_dataset, indent=4)

        with open("data/splits/cn-fin-ari.json", "w") as outfile:
            outfile.write(json_object)
