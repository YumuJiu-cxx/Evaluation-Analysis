import re


def remove_punctuation(line):
    """删除除字母、数字、汉字以外的所有符号"""
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)

    return line


def stopwordslist(filepath):
    """删除停用词"""
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]

    return stopwords
