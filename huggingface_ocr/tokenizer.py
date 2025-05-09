import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import alphabet
class Tokenizer:
    def __init__(self, token_id_dict):
        # Special tokens
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        # 27 is space but since we are not using that one is subtracted from len of token_id_dict
        token_id_dict.update({
            self.pad_token: len(token_id_dict) - 1,
            self.bos_token: len(token_id_dict),
            self.eos_token: len(token_id_dict) + 1
        })

        self.token_to_id = token_id_dict
        self.id_to_token = alphabet.enum_to_hebrew
        
        # IDs for convenience
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
    
    def encode(self, text):
        # Turn string into list of token IDs
        tokens = [self.token_to_id[c] for c in text]
        return self.add_control_tokens(tokens)
    
    def add_control_tokens(self, tokens):
        return np.concatenate([np.array([self.bos_token_id]), tokens, np.array([self.eos_token_id])])
    
    def id_to_hebrew_char(self, id: int) -> str:
        enum_member = next((k for k, v in alphabet.char_token.items() if v == id), None)
        if enum_member is None:
            return ''
        return alphabet.enum_to_hebrew.get(enum_member, '')

    def decode(self, ids):
        # Turn list of IDs into string, ignore special tokens
        tokens = [self.id_to_hebrew_char(i) for i in ids if i not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
        return ''.join(tokens)
    
    def vocab_size(self):
        return len(self.token_to_id)