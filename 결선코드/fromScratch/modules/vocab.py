import csv
import pandas as pd

class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


class KoreanSpeechVocabulary(Vocabulary):
    def __init__(self, 
                 vocab_path, 
                 output_unit: str = 'character', 
                 sp_model_path=None, #??
                 sos_id = '<s>',
                 eos_id = '</s>',
                 pad_id = '[pad]',
                 blank_id = '<blank>',
                 if_add_blank_id = False
                #  unk_id = '[unk]'
                 ):
        super(KoreanSpeechVocabulary, self).__init__()
        
        vocab_df = pd.read_csv(vocab_path)
        vocab_df['char'][vocab_df['char'] == '<sos>'] = sos_id
        vocab_df['char'][vocab_df['char'] == '<eos>'] = eos_id
        vocab_df['char'][vocab_df['char'] == '<pad>'] = pad_id
        self.if_add_blank_id = if_add_blank_id
        if if_add_blank_id:
            self.blank = blank_id
        # vocab_df['char'][vocab_df['char'] == '<blank>'] = blank_id # load_vocab에서 black를 만드네

        # try:
        #     vocab_df['char'][vocab_df['char'] == '[unk]'] = unk_id
        # except:
        #     vocab_df = vocab_df.append(pd.DataFrame([len(vocab_df), unk_id, 0], columns = ['id','char','freq']))
        vocab_df.to_csv(vocab_path.split(",")[0] + '_edit.csv', index=False)

        self.vocab_dict, self.id_dict = self.load_vocab(vocab_path.split(",")[0] + '_edit.csv', encoding='utf-8')
        self.sos_id = int(self.vocab_dict[sos_id])
        self.eos_id = int(self.vocab_dict[eos_id])
        self.pad_id = int(self.vocab_dict[pad_id])
        if if_add_blank_id:
            self.blank_id = int(self.vocab_dict[blank_id])
        # self.unk_id = int(self.vocab_dict[unk_id])
        self.labels = self.vocab_dict.keys()

        self.vocab_path = vocab_path
        self.output_unit = output_unit

    def __len__(self):

        return len(self.vocab_dict)

    def label_to_string(self, labels):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """

        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
        """
        Provides char2id, id2char

        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method

        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()

        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

            if self.if_add_blank_id:
                unit2id[self.blank] = len(unit2id)
                id2unit[len(unit2id)] = self.blank

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))