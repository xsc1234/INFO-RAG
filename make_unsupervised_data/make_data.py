import csv
import random
import logging
logger = logging.getLogger(__name__)
from nltk.tokenize import sent_tokenize
import joblib,time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--psgs_data_path',
                        type=str,
                        default='/tmp/data_files/')
    parser.add_argument(
        '--uns_data_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Save uns data path'
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    psgs_data_path = args.psgs_data_path
    save_data_path = args.uns_data_path
    logger.info("Loading Corpus DPR wiki...")
    with open(psgs_data_path, 'r', encoding='utf-8-sig') as r:
        reader = csv.reader(r, delimiter="\t")
        corpus_list = []
        load_num = 0
        for row in tqdm(reader):
            corpus_list.append(row)
            load_num += 1
            if load_num > 8000000:
                break

        sub_corpus = corpus_list[:8000000] #Select 800w passages

        count = 0
        corpus = {}
        title_now = ''
        text_now = ''
        train_query = []
        qrels = []
        sentence_count = 0
        title_id_count = 0
        start_time = time.time()
        passage_list = []
        res_list = []
        for row in tqdm(sub_corpus):
            if count % 10000 == 0:
                end_time = time.time()
                print('{}count {}s'.format(count,end_time-start_time))
            if count == 0:
                count += 1
                continue
            if count == 1:
                title_now = row[2]
                text_now = row[1]
                passage_list.append((row[0],row[1]))

            if row[2] == title_now and len(passage_list) < 5:
                text_now += (' ' + row[1])
                passage_list.append((row[0],row[1]))
            else:
                sentences = sent_tokenize(text_now) # Split the article into sentences
                for i in range(len(sentences)):
                    sentences[i] = title_now + '. ' + sentences[i]
                random_sentence = random.choice(sentences)
                for i in range(len(sentences)):
                    if random_sentence == sentences[i]:
                        juli_score_list = []
                        for j in range(len(sentences)):
                            juli_score = abs(i-j)
                            juli_score_list.append(juli_score)

                        data = []
                        data[0] = str(count)

                        #selected = random_sentence
                        #passage_list = sentences
                        #score = juli_score_list
                        #data clearner
                        score_new = []
                        passage_list_new = []
                        selected_tag = False
                        for i in range(len(sentences)):
                            if sentences[i] == random_sentence and (not selected_tag) and (not juli_score_list[i] == 0):
                                continue
                            if sentences[i] == random_sentence and not selected_tag:
                                selected_tag = True
                                temp = sentences[i].split('. ')[1:]
                                temp_str = 'number i'.format(i)
                                for t in temp:
                                    temp_str = temp_str + ' ' + t
                                passage_list_new.append(temp_str)
                                data[1] = temp_str
                                score_new.append(0)
                            elif sentences[i] == random_sentence and selected_tag:
                                continue
                            else:
                                temp = sentences[i].split('. ')[1:]
                                temp_str = 'number i'.format(i)
                                for t in temp:
                                    temp_str = temp_str + ' ' + t
                                passage_list_new.append(temp_str)
                                score_new.append(juli_score_list[i])
                        data[3] = score_new
                        data[2] = passage_list_new
                        res_list.append(data)

                title_id_count += 1
                title_now = row[2]
                text_now = row[1]
                passage_list = [(row[0], row[1])]
            count += 1
        print('done!!!')
        joblib.dump(res_list, save_data_path)