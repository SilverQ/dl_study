"""
Break the Caesar!
Instructions
    The Caesar cipher is a notorious (and notoriously simple) algorithm for encrypting a message:
    each letter is shifted a certain constant number of places in the alphabet.
    For example, applying a shift of 5 to the string "Hello, world!" yields "Mjqqt, btwqi!", which is jibberish.
    In this kata, your task is to decrypt Caesar-encrypted messages using nothing but your wits, your computer,
    and a set of the 1000 (plus a few) most common words in English in lowercase (made available to you as a preloaded
    variable named WORDS, which you may use in your code as if you had defined it yourself).
    Given a message, your function must return the most likely shift value as an integer.
A few hints:
    Be wary of punctuation, Shift values may not be higher than 25
"""
import nltk

WORDS = {'finger', 'way', 'last', 'ball', 'house', 'pick', 'division', 'interest', 'baby', 'many', 'wife', 'came',
         'speak', 'garden', 'rail', 'box', 'spell', 'ring', 'order', 'after', 'should', 'new', 'operate', 'face',
         'fly', 'wind', 'mind', 'when', 'through', 'figure', 'dry', 'leg', 'want', 'written', 'well', 'raise',
         'length', 'answer', 'ocean', 'cotton', 'works', 'pose', 'third', 'went', 'quotient', 'dream', 'own',
         'square', 'better', 'yard', 'protect', 'brought', 'the', 'fat', 'sleep', 'bird', 'to', 'would', 'sentence',
         'expect', 'sit', 'high', 'basic', 'family', 'yet', 'where', 'voice', 'weather', 'sell', 'flat', 'out', 'cent',
         'music', 'fear', 'question', 'broad', 'machine', 'more', 'fact', 'nose', 'port', 'guide', 'until', 'select',
         'quick', 'an', 'board', 'office', 'let', 'blood', 'sister', 'fun', 'blow', 'feed', 'whose', 'land', 'meet',
         'poor', 'quite', 'truck', 'art', 'knew', 'evening', 'gas', 'cost', 'large', 'morning', 'add', 'together',
         'chick', 'think', 'week', 'pull', 'hill', 'chair', 'earth', 'die', 'street', 'flower', 'enemy', 'hot', 'by',
         'indicate', 'atom', 'must', 'seven', 'saw', 'help', 'possible', 'got', 'claim', 'molecule', 'bottom', 'which',
         'four', 'winter', 'noun', 'fig', 'clear', 'support', 'before', 'character', 'clothe', 'about', 'hour',
         'village', 'try', 'rose', "don't", 'were', 'note', 'person', 'money', 'quart', 'their', 'lost', 'point',
         'either', 'this', 'least', 'happen', 'practice', 'will', 'depend', 'tree', 'girl', 'city', 'gentle', 'twenty',
         'might', 'able', 'long', 'call', 'string', 'deal', 'climb', 'stream', 'believe', 'care', 'hold', 'material',
         'cow', 'period', 'tell', 'yellow', 'meat', 'simple', 'he', 'center', 'page', 'steam', 'list', 'down', 'race',
         'effect', 'on', 'party', 'then', 'product', 'hunt', 'noise', 'anger', 'rather', 'matter', 'it', 'sent', 'run',
         'wide', 'study', 'perhaps', 'substance', 'correct', 'could', 'pay', 'east', 'us', 'poem', 'fill', 'compare',
         'follow', 'tall', 'next', 'left', 'equal', 'began', 'turn', 'pass', 'song', 'second', 'cloud', 'wrong', 'sun',
         'century', 'age', 'wtf', 'problem', 'lake', 'summer', 'sing', 'has', 'rich', 'window', 'duck', 'busy', 'row',
         'reason', 'north', 'term', 'result', 'describe', 'post', 'example', 'five', 'hand', 'first', 'phrase', 'stand',
         'chart', 'miss', 'end', 'tone', 'over', 'while', 'sudden', 'even', 'corn', 'salt', 'bat', 'hello', 'real',
         'book', 'fair', 'chance', 'inch', 'mine', 'moment', 'green', 'never', 'value', 'spoke', 'require', 'famous',
         'every', 'check', 'temperature', 'bit', 'bell', 'motion', 'record', 'collect', 'star', 'men', 'road', 'deep',
         'wrote', 'day', 'spread', 'been', 'ten', 'jump', 'feet', 'vary', 'now', 'coat', 'particular', 'drive',
         'soldier', 'friend', 'favor', 'back', 'oh', 'young', 'sail', 'root', 'kept', 'agree', 'win', 'whole', 'seed',
         'free', 'repeat', 'populate', 'shop', 'consider', 'use', 'she', 'travel', 'enter', 'several', 'search', 'ask',
         'these', 'forest', 'if', 'is', 'continent', 'lot', 'skin', 'with', 'school', 'coast', 'caught', 'made', 'sand',
         'hat', 'parent', 'gray', 'some', 'mix', 'work', 'liquid', 'spring', 'off', 'speed', 'warm', 'law', 'short',
         'leave', 'begin', 'captain', 'rub', 'experiment', 'wear', 'seat', 'rain', 'radio', 'bar', 'season', 'major',
         'log', 'among', 'animal', 'tail', 'object', 'any', 'come', 'ease', 'crop', 'three', 'ear', 'bought', 'few',
         'water', 'bright', 'element', 'low', 'measure', 'jumps', 'grew', 'instant', 'certain', 'column', 'see', 'oil',
         'line', 'mile', 'part', 'start', 'people', 'exercise', 'edge', 'view', 'walk', 'plane', 'heavy', 'ground',
         'meant', 'choose', 'me', 'eye', 'general', 'war', 'held', 'shoulder', 'carry', 'middle', 'build', 'seem',
         'wall', 'snow', 'test', 'fight', 'master', 'get', 'arrive', 'pitch', 'only', 'or', 'town', 'design',
         'thought', 'moon', 'ever', 'front', 'close', 'appear', 'broke', 'prepare', 'single', 'egg', 'camp', 'decimal',
         'oxygen', 'roll', 'stop', 'of', 'body', 'blue', 'silent', 'horse', 'neighbor', 'foot', 'count', 'teeth',
         'soon', 'light', 'dress', 'much', 'room', 'nine', "won't", 'silver', 'subject', 'shore', 'supply', 'big',
         'lazy', 'language', 'full', 'govern', 'fit', 'lead', 'difficult', 'rule', 'circle', 'total', 'life', 'mark',
         'level', 'determine', 'produce', 'pound', 'nation', 'right', 'visit', 'flow', 'father', 'heat', 'go', 'hit',
         'cross', 'field', 'paint', 'son', 'home', 'save', 'condition', 'press', 'class', 'took', 'found', 'human',
         'process', 'allow', 'fraction', 'decide', 'bread', 'special', 'kind', 'very', 'charge', 'ago', 'done',
         'system', 'card', 'gone', 'plural', 'suggest', 'experience', 'slow', 'door', 'crease', 'six', 'reach',
         'common', 'necessary', 'plan', 'sign', 'remember', 'gave', 'sugar', 'shout', 'receive', 'present', 'hear',
         'kill', 'insect', 'again', 'double', 'trip', 'thin', 'no', 'plant', 'distant', 'push', 'divide', 'wood',
         'slave', 'store', 'fish', 'two', 'represent', 'tiny', 'pair', 'near', 'half', 'finish', 'my', 'listen',
         'multiply', 'behind', 'grow', 'control', 'heard', 'prove', 'happy', 'once', 'hair', 'fresh', 'trouble',
         'success', 'lift', 'look', 'separate', 'send', 'mass', 'thus', 'open', 'range', 'have', 'electric', 'bank',
         'play', 'cut', 'iron', 'market', 'dear', 'region', 'surface', 'position', 'at', 'act', 'track', 'story', 'buy',
         'planet', 'mean', 'opposite', 'glad', 'a', 'sheet', 'metal', 'differ', 'quiet', 'bad', 'does', 'car', 'stead',
         'farm', 'numeral', 'feel', 'type', 'here', 'cook', 'in', 'between', 'boy', 'imagine', 'top', 'vowel', 'food',
         'hope', 'strange', 'find', 'tire', 'bring', 'ship', 'black', 'suit', 'afraid', 'current', 'born', 'area',
         'far', 'ran', 'can', 'gun', 'glass', 'great', 'melody', 'organ', 'said', 'form', 'country', 'often', 'soil',
         'ice', 'but', 'clean', 'was', 'laugh', 'fine', 'west', 'as', 'just', 'little', 'sight', 'arm', 'from', 'drop',
         'fox', 'shell', 'break', 'cause', 'settle', 'his', 'above', 'magnet', 'enough', 'history', 'soft', 'women',
         'serve', 'danger', 'path', 'children', 'rope', 'student', 'told', 'wish', 'may', 'dead', 'true', 'company',
         'brother', 'thank', 'draw', 'air', 'shape', 'throw', 'island', 'other', 'section', 'that', 'please', 'space',
         'locate', 'stood', 'love', 'round', 'smile', 'under', 'same', 'ready', 'table', 'we', 'million', 'trade',
         'ride', 'shoe', 'like', 'energy', 'skill', 'had', 'catch', 'reply', 'clock', 'river', 'unit', 'similar',
         'exact', 'sat', 'her', 'always', 'for', 'are', 'strong', 'match', 'late', 'cover', 'smell', 'game', 'give',
         'except', 'hard', 'learn', 'number', 'develop', 'stick', 'idea', 'contain', 'team', 'dad', 'wonder', 'valley',
         'read', 'solve', 'too', 'excite', 'spend', 'chief', 'map', 'take', 'lay', 'triangle', 'felt', 'print', 'fire',
         'hundred', 'weight', 'job', 'mountain', 'especially', 'red', 'industry', 'beat', 'neck', 'dark', 'thousand',
         'bone', 'mount', 'hurry', 'modern', 'share', 'thing', 'world', 'method', 'them', 'up', 'nothing', 'you',
         'copy', 'make', 'sea', 'guess', 'power', 'case', 'equate', 'arrange', 'crowd', 'set', 'most', 'tube', 'grass',
         'syllable', 'what', 'segment', 'one', 'group', 'occur', 'move', 'noon', 'joy', 'know', 'suffix', 'else',
         'fruit', 'event', 'safe', 'drink', 'than', 'such', 'stay', 'stone', 'engine', 'cool', 'gather', 'proper',
         'consonant', 'science', 'boat', 'write', 'apple', 'him', 'show', 'natural', 'each', 'cat', 'do', 'lie', 'live',
         'join', 'be', 'led', 'solution', 'colony', 'rise', 'paragraph', 'doctor', 'self', 'who', 'corner', 'both',
         'loud', 'lady', 'plain', 'teach', 'and', 'dollar', 'talk', 'speech', 'dog', 'fell', 'watch', 'early', 'letter',
         'your', 'mother', 'still', 'usual', 'white', 'though', 'course', 'keep', 'burn', 'sky', 'man', 'place', 'wire',
         'continue', 'shine', 'desert', 'direct', 'cry', 'need', 'dictionary', 'death', 'say', 'rock', 'fast', 'size',
         'put', 'word', 'score', 'include', 'step', 'lone', 'huge', 'wave', 'am', 'forward', 'force', 'gold', 'minute',
         'sharp', 'cold', 'small', 'paper', 'shall', 'stretch', 'fall', 'nor', 'did', 'why', 'provide', 'woman', 'eat',
         'observe', 'station', 'offer', 'old', 'instrument', 'probable', 'tie', 'connect', 'year', 'they', 'swim',
         'grand', 'degree', 'change', 'less', 'verb', 'capital', 'invent', 'month', 'our', 'piece', 'hole', 'picture',
         'create', 'state', 'so', 'there', 'past', 'night', 'color', 'side', 'wheel', 'bear', 'best', 'sure',
         'straight', 'original', 'notice', 'rest', 'main', 'key', 'scale', 'yes', 'against', 'wild', 'touch', 'head',
         'all', 'during', 'surprise', 'subtract', 'tool', 'nature', 'time', 'south', 'symbol', 'good', 'whether',
         'wing', 'eight', 'pretty', 'wash', 'chord', 'child', 'pattern', 'discuss', 'bed', 'steel', 'spot', 'those',
         'dance', 'floor', 'name', 'milk', 'base', 'brown', 'cell', 'slip', 'train', 'how', 'block', 'branch', 'also',
         'since', 'toward', 'thick', 'sound', 'wait', 'final', 'sense', 'property', 'band', 'king', 'beauty',
         'complete', 'mouth', 'heart'}
# print(WORDS)
lst = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
# print('lst: ', lst)


def conv_str(c, i):
    return lst[lst.index(c) - i if lst.index(c) - i < len(lst) else lst.index(c) - i - 52 * ((lst.index(c) - i) // 52)]


def break_caesar(message):
    splited_message = message.split(' ')
    cleaned_message = [''.join([c for c in word if c in lst]) for word in splited_message]
    # print(cleaned_message)
    # for i in range(26):
    #     decoded_message = [''.join([conv_str(c, i) for c in word]) for word in cleaned_message]
    decoded_message = [[''.join([conv_str(c, i) for c in word]) for word in cleaned_message] for i in range(26)]
    # print(decoded_message)
    matching = [sum(1 if word.lower() in WORDS else 0 for word in line) for line in decoded_message]
    # for line in decoded_message:
    #     print(line)
    #     chk = 0
    #     for word in line:
    #         print(word.lower(), word.lower() in WORDS)
    #         if word.lower() in lst:
    #             chk += 1
    #         print(word, chk)
    result = matching.index(max(matching))
    return result


def break_caesar_old(message):
    for i in range(26):
        decoded_message = ''.join([conv_str(c, i) if c in lst else c for c in message])
        print(decoded_message)
        words = decoded_message.split(' ')
        chk = 0
        for word in words:
            word = ''.join([c for c in word if c in lst])
            chk += 1 if word.lower() in WORDS else 0
        print(words, i, chk)
        # if chk == len(words):
        if chk > 0:
            print(words, i, chk, len(words))
            # return i
    # return  # the most likely shift value as an integer


# test.assert_equals(break_caesar("DAM? DAM! DAM."), 7)
# test.assert_equals(break_caesar("Mjqqt, btwqi!"), 5)
# test.assert_equals(break_caesar("Gur dhvpx oebja sbk whzcf bire gur ynml qbt."), 13)

# print(break_caesar("DAM? DAM! DAM."))      # 7
# print(break_caesar("Mjqqt, btwqi!"))       # 5
# print(break_caesar("Gur dhvpx oebja sbk whzcf bire gur ynml qbt."))        # 13
# print(break_caesar('Czvo gtkwjnw fih, rz ajmh cjmnz azgo odnokd yzxdyz czvm jpo izxznnvmt adivg, cdo oxg lbgz roa yzkgj, zsxdoz, gvfz ndbi hwt pil qbmqoi epno gvibpvbz ngvqz kmznzio moaaymn vit kvmvbmvkc npbvm cjgz wvy wmjri orziot avqjm wjs bekt hjno rvt hja?'))
print(break_caesar('Czvo gtkwjnw fih, rz ajmh cjmnz azgo odnokd yzxdyz czvm jpo izxznnvmt adivg'))


# Other's Answers
# anter69
abc = 'abcdefghijklmnopqrstuvwxyz'
def caesar(s, shift):
    # make a translation table with the current shift
    transtable = str.maketrans(abc, abc[shift:] + abc[:shift])
    return s.translate(transtable)
def break_caesar(message):
    # sanitize the input
    message = ''.join(c if c.isalpha() else ' ' for c in message).lower()
    # keep track of hits
    hits = []
    # try all possible shifts
    for shift in range(26):
        # decode the message with the current shift
        decoded = caesar(message, -shift)
        cnt = 0
        for word in decoded.split():
            # count the number of common English words
            if word in WORDS:
                cnt += 1
        # append the result
        hits.append(cnt)
    # find the most likely shift value
    shift = hits.index(max(hits))
    return shift

# anter69
abc = 'abcdefghijklmnopqrstuvwxyz'
def break_caesar(message):
    caesar = lambda s, key: s.translate(str.maketrans(abc, abc[key:] + abc[:key]))
    message = ''.join(c if c.isalpha() else ' ' for c in message).lower()
    hits = [ sum(w in WORDS for w in caesar(message, -key).split()) for key in range(26) ]
    return hits.index(max(hits))


# Blind4Basics
from string import ascii_lowercase as low
import re
def break_ceasar(msg):
    return max((len(WORDS & set(re.split(r'[^\w]', msg.lower().translate(str.maketrans(low, low[26-n:]+low[:26-n]))))), n) for n in range(26))[1]


# lechevalier
import re;break_caesar=lambda s:max(range(26),key=lambda n:len({''.join(chr((ord(c)-97-n)%26+97)for c in w)for w in re.findall('[a-z]+',s.lower())}&WORDS))

