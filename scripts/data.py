import math
import gzip
import json
from itertools import accumulate
import re
import torch
import torch.utils.data as td


class DataSet(td.Dataset):

    @classmethod
    def from_files(cls, fnames, reference=None, limit=None, zeroing_proportion=0.0):
        if not isinstance(fnames, list):
            fnames = [fnames]
        
        items = []
        for fname in fnames:
            with gzip.open(fname) as ifd:
                for n, line in enumerate(ifd):
                    if len(items) == limit:
                        break                    
                    item = json.loads(line)
                    item["text"] = re.sub(r"\s+", " ", item["text"])
                    words = item["text"].split()
                    word_scans = item["scansion"].split()
                    items.append(
                        {
                            "characters" : item["text"],
                            "words" : words,
                            "scans" : word_scans,
                            "stresses" : [[s for s in sp] for sp in word_scans],
                            "length_in_words" : len(words),
                            "length_in_characters" : len(item["text"]),
                            "word_end_character_indices" : list(accumulate([len(x) + (1 if i > 0 else 0) for i, x in enumerate(words)], initial=-1))[1:],
                            "scan_lengths" : [len(s) for s in word_scans],
                            "index" : len(items),
                            "original" : item,
                        }
                    )
        return DataSet(items, reference=reference, zeroing_proportion=zeroing_proportion)

    
    def __init__(self, items, reference=None, zeroing_proportion=0.0):
        
        super(DataSet, self).__init__()
        
        self._items = items
        self._length = len(items)
        self._max_words_per_line = max([len(x["words"]) for x in items])
        self._max_chars_per_line = max([len(x["characters"]) for x in items])
        # one more extra for "end of sequence"
        self._max_stresses_per_word = max([len(s) for s in sum([x["stresses"] for x in items], [])]) + 1

        self._word2id = reference._word2id if reference else {None : 0}
        self._char2id = reference._char2id if reference else {None : 0}
        self._scan2id = reference._scan2id if reference else {None : 0}
        # one more extra for "end of sequence"
        self._stress2id = reference._stress2id if reference else {None : 0, True : 1}
        
        # characters: length x max_chars_per_line
        # words: length x max_words_per_line
        # scans: length x max_words_per_line
        # stresses: length x max_words_per_line x max_stresses_per_word
        
        self._characters = torch.zeros((self._length, self._max_chars_per_line), device="cpu", dtype=torch.int64)
        self._char_rand = torch.rand_like(self._characters, dtype=torch.float)
        self._words = torch.zeros((self._length, self._max_words_per_line), device="cpu", dtype=torch.int64)
        self._word_rand = torch.rand_like(self._words, dtype=torch.float)
        self._scans = torch.zeros((self._length, self._max_words_per_line), device="cpu", dtype=torch.int64)
        self._stresses = torch.zeros((self._length, self._max_words_per_line, self._max_stresses_per_word + 1), device="cpu", dtype=torch.int64)
        self._aligned = torch.full((self._length,), fill_value=False, device="cpu", dtype=torch.bool)
        self._length_in_words = torch.zeros(self._length, device="cpu", dtype=torch.int64)
        self._word_length_in_stresses = torch.zeros((self._length, self._max_words_per_line), device="cpu", dtype=torch.int64)
        self._scan_mask = torch.full(self._words.shape, fill_value=False, device="cpu", dtype=torch.bool)
        self._stress_mask = torch.full(self._stresses.shape, fill_value=False, device="cpu", dtype=torch.bool)
        self._word_end_character_mask = torch.full(self._characters.shape, fill_value=False, device="cpu", dtype=torch.bool)
        self._word_mask = torch.full(self._words.shape, fill_value=False, device="cpu", dtype=torch.bool)
        self._indices = torch.zeros(self._length, device="cpu", dtype=torch.int64)
        
        for i, item in enumerate(self._items):
            self._indices[i] = i
            self._aligned[i] = len(item["words"]) == len(item["scans"])
            self._length_in_words[i] = len(item["words"])
            for c, char in enumerate(item["characters"]):
                if char not in self._char2id and not reference:
                    self._char2id[char] = len(self._char2id)
                self._characters[i, c] = self._char2id.get(char, 0)

            for w, (word, end) in enumerate(zip(item["words"], item["word_end_character_indices"])):
                
                if word not in self._word2id and not reference:
                    self._word2id[word] = len(self._word2id)
                self._words[i, w] = self._word2id.get(word, 0)
                self._word_mask[i, w] = True
                self._word_end_character_mask[i, end] = True                
                if self._aligned[i]:
                    self._scan_mask[i, w] = True
                    sc = item["scans"][w]
                    if sc not in self._scan2id and not reference:
                        self._scan2id[sc] = len(self._scan2id)                    
                    self._scans[i, w] = self._scan2id.get(sc, 0)
                    for s, st in enumerate(item["stresses"][w]):
                        if st not in self._stress2id and not reference:
                            self._stress2id[st] = len(self._stress2id)
                        self._stresses[i, w, s] = self._stress2id.get(st, 0)
                        self._stress_mask[i, w, s] = True
                    # account for "end-of-word-scan" value of 1
                    self._word_length_in_stresses[i, w] = len(item["stresses"][w]) + 1
                    self._stresses[i, w, len(item["stresses"][w])] = 1
                    self._stress_mask[i, w, len(item["stresses"][w])] = True
                    
        if zeroing_proportion > 0.0:
            self._words[self._word_rand < zeroing_proportion] = 0
            
        self._id2word = {v : k for k, v in self._word2id.items()}
        self._id2char = {v : k for k, v in self._char2id.items()}
        self._id2scan = {v : k for k, v in self._scan2id.items()}
        self._id2stress = {v : k for k, v in self._stress2id.items()}
        
    def to(self, device):
        self._characters = self._characters.to(device)
        self._words = self._words.to(device)
        self._scans = self._scans.to(device)
        self._stresses = self._stresses.to(device)
        self._aligned = self._aligned.to(device)
        self._word_length_in_stresses = self._word_length_in_stresses.to(device)
        self._scan_mask = self._scan_mask.to(device)
        self._stress_mask = self._stress_mask.to(device)
        self._word_mask = self._word_mask.to(device)
        self._word_end_character_mask = self._word_end_character_mask.to(device)
        self._indices = self._indices.to(device)
        self._length_in_words = self._length_in_words.to(device)
        return self

    def nwords(self):
        return len(self._id2word)

    def nscans(self):
        return len(self._id2scan)

    def nstresses(self):
        return len(self._id2stress)

    def nchars(self):
        return len(self._id2chars)
    
    def __len__(self):
        return self._length


class DataLoader(object):

    def __init__(self, dataset, batch_size, shuffle=False, device="cpu"):
        self._device = device
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._indices = torch.arange(len(self._dataset), device="cpu")
        self._num_batches = math.ceil(len(self._dataset) / self._batch_size)

    def __iter__(self):
        indices = torch.randperm(len(self._dataset), device=self._device) if self._shuffle else torch.arange(len(self._dataset), device=self._device)
        for b in range(self._num_batches):
            start = b * self._batch_size
            end = min((b + 1) * self._batch_size, len(self._dataset))
            idx = indices[start:end]
            yield {
                "words" : self._dataset._words[idx],
                "characters" : self._dataset._characters[idx],
                "scans" : self._dataset._scans[idx],
                "stresses" : self._dataset._stresses[idx],
                "aligned" : self._dataset._aligned[idx],
                "length_in_words" : self._dataset._length_in_words[idx],
                "word_length_in_stresses" : self._dataset._word_length_in_stresses[idx],
                "scan_mask" : self._dataset._scan_mask[idx],
                "stress_mask" : self._dataset._stress_mask[idx],
                "word_end_character_mask" : self._dataset._word_end_character_mask[idx],
                "word_mask" : self._dataset._word_mask[idx],
                #"word_rand" : self._dataset._word_rand[idx],
                #"char_rand" : self._dataset._char_rand[idx],                
                "indices" : self._dataset._indices[idx]
            }
        return None
