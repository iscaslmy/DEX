# 数据对象的类
import re

import jieba


class DataSample:

    def __init__(self,
                 story_id,
                 sentence,
                 fps=[],
                 p=0.0):
        self.story_id = story_id
        self.sentence = sentence
        self.sen_words = jieba.lcut(sentence)
        self.fps = fps
        self.p = p
        self.label = self.get_label(self.sentence)
        self.char_label = self.get_char_label()

    def get_label(self, sentence):
        encode = ["N" for _ in range(len(self.sen_words))]
        fps = self.fps
        if fps is None or len(fps) == 0 or fps[0] == 'null':
            return encode
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                if fps[j].__contains__(fps[i]):
                    fps[i] = ''
                if fps[i].__contains__(fps[j]):
                    fps[j] = ''
        for fp in fps:
            try:
                if fp == '':
                    continue
                fp_words = jieba.lcut(fp)
                fp_tmp = re.sub('[\+]', '\\+', fp)
                fp_tmp = re.sub('[\(]', '\\(', fp_tmp)
                starts = [i.start() for i in re.finditer(fp_tmp, sentence)]
                ends = [i + len(fp) - 1 for i in starts]
                for start, end in zip(starts, ends):
                    start_idx = 0
                    end_idx = 0
                    for s in self.sen_words:
                        if start >= len(s):
                            start -= len(s)
                            start_idx += 1
                        else:
                            break
                    for s in self.sen_words:
                        if end >= len(s):
                            end -= len(s)
                            end_idx += 1
                        else:
                            break
                    try:
                        if len(fp_words) == 1:
                            encode[start_idx] = "S"
                        elif len(fp_words) == 2:
                            encode[start_idx] = "B"
                            encode[end_idx] = "E"
                        elif len(fp_words) > 2:
                            encode[start_idx] = "B"
                            encode[end_idx] = "E"
                            for k in range(start_idx + 1, end_idx):
                                encode[k] = "I"
                    except IndexError:
                        print(fp_words, self.sen_words, start_idx, end_idx)

            except Exception:
                continue
        return encode

    def get_char_label(self):
        encode = ['N' for _ in range(len(list(self.sentence)))]
        fps = self.fps
        if fps is None or len(fps) == 0 or fps[0] == 'null':
            return encode
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                if fps[j].__contains__(fps[i]):
                    fps[i] = ''
                if fps[i].__contains__(fps[j]):
                    fps[j] = ''
        for fp in fps:
            try:
                if fp == '':
                    continue
                fp_tmp = re.sub('[\+]', '\\+', fp)
                fp_tmp = re.sub('[\(]', '\\(', fp_tmp)
                starts = [i.start() for i in re.finditer(fp_tmp, self.sentence)]
                ends = [i + len(fp) - 1 for i in starts]
                for start, end in zip(starts, ends):
                    if len(fp) == 1:
                        encode[start] = "S"
                    else:
                        encode[start] = "B"
                        encode[start + len(fp) - 1] = "E"
                        for k in range(1, len(fp) - 1):
                            encode[start + k] = "I"
            except Exception:
                continue
        return encode
