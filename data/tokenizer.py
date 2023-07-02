from vocabulary.tokenizer import TransformerTokenizer
from vocabulary.gen_vocab import reserved_tokens


class Tokenizer:

    def __init__(self, data_dir='./data/'):
        self.data_dir = data_dir

    def get_ted_tokenizer(self):
        en_ted = self.data_dir + 'en_ted_vocab.txt'
        pt_ted = self.data_dir + 'pt_ted_vocab.txt'
        en_tokenizer = TransformerTokenizer(en_ted, res_tokens=reserved_tokens)
        pt_tokenizer = TransformerTokenizer(pt_ted, res_tokens=reserved_tokens)
        return pt_tokenizer, en_tokenizer
    
    def get_encorp_tokenizer(self):
        en_ted = self.data_dir + 'en_vocab_encorp.txt'
        pt_ted = self.data_dir + 'hindi_vocab_encorp.txt'
        en_tokenizer = TransformerTokenizer(en_ted, res_tokens=reserved_tokens)
        hi_tokenizer = TransformerTokenizer(pt_ted, res_tokens=reserved_tokens)
        return en_tokenizer, hi_tokenizer
