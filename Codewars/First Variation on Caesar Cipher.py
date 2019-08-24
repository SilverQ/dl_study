# -*- coding: utf-8 -*-
from nose.tools import assert_equals

''' Instruction
The action of a Caesar cipher is to replace each plaintext letter with a different one a fixed number of places up or down the alphabet.
This program performs a variation of the Caesar shift. The shift increases by 1 for each character (on each iteration).
If the shift is initially 1, the first character of the message to be encoded will be shifted by 1, the second character will be shifted by 2, etc...

Coding: Parameters and return of function "movingShift"
param s: a string to be coded
param shift: an integer giving the initial shift

The function "movingShift" first codes the entire string and then returns an array of strings containing the coded string in 5 parts (five parts because, to avoid more risks, the coded message will be given to five runners, one piece for each runner).
If possible the message will be equally divided by message length between the five runners. If this is not possible, parts 1 to 5 will have subsequently non-increasing lengths, such that parts 1 to 4 are at least as long as when evenly divided, but at most 1 longer. If the last part is the empty string this empty string must be shown in the resulting array.
For example, if the coded message has a length of 17 the five parts will have lengths of 4, 4, 4, 4, 1. The parts 1, 2, 3, 4 are evenly split and the last part of length 1 is shorter. If the length is 16 the parts will be of lengths 4, 4, 4, 4, 0. Parts 1, 2, 3, 4 are evenly split and the fifth runner will stay at home since his part is the empty string. If the length is 11, equal parts would be of length 2.2, hence parts will be of lengths 3, 3, 3, 2, 0.
You will also implement a "demovingShift" function with two parameters

Decoding: parameters and return of function "demovingShift"
1) an array of strings: s (possibly resulting from "movingShift", with 5 strings)
2) an int shift
"demovingShift" returns a string.

Example:
u = "I should have known that you would have a perfect answer for me!!!"
movingShift(u, 1) returns :
v = ["J vltasl rlhr ", "zdfog odxr ypw", " atasl rlhr p ", "gwkzzyq zntyhv", " lvz wp!!!"]
(quotes added in order to see the strings and the spaces, your program won't write these quotes, see Example Test Cases)
and demovingShift(v, 1) returns u.
'''

#     print('a: {}, z: {}, A: {}, Z: {}'.format(ord('a'), ord('z'), ord('A'), ord('Z'))), a: 97, z: 122, A: 65, Z: 90


def moving_shift(s, shift):
    if len(s) % 5 == 0:
        split_len = len(s) // 5
    else:
        split_len = len(s) // 5 + 1
    results_chr = []
    for j in range(5):
        results_chr_temp = []
        splited_s = s[split_len*j:split_len*(j + 1)]
        for i, c in enumerate(splited_s):
            if c.isupper():
                s_shift = (ord(c) + shift + i + j* split_len - 65) % 26 + 65
            elif c.islower():
                s_shift = (ord(c) + shift + i + j * split_len - 97) % 26 + 97
            else:
                s_shift = ord(c)
            results_chr_temp.append(chr(s_shift))
        results_chr.append(''.join(results_chr_temp))
    return results_chr


def demoving_shift(s, shift):
    split_len = len(s[0])
    #     print('split length: ', split_len)
    results_chr = []
    for j in range(5):
        splited_s = s[j]
        for i, c in enumerate(splited_s):
            if c.isupper():
                s_shift = (ord(c) - shift - i - j * split_len - 65) % 26 + 65
            elif c.islower():
                s_shift = (ord(c) - shift - i - j * split_len - 97) % 26 + 97
            else:
                s_shift = ord(c)
            results_chr.append(chr(s_shift))
    return ''.join(results_chr)


# test.assert_equals(
#     moving_shift("I should have known that you would have a perfect answer for me!!!", 1),
#     ["J vltasl rlhr ", "zdfog odxr ypw", " atasl rlhr p ", "gwkzzyq zntyhv", " lvz wp!!!"])
#
# test.assert_equals(
#     demoving_shift(["J vltasl rlhr ", "zdfog odxr ypw", " atasl rlhr p ", "gwkzzyq zntyhv", " lvz wp!!!"], 1),
#     "I should have known that you would have a perfect answer for me!!!")

assert_equals(
    moving_shift("I should have known that you would have a perfect answer for me!!!", 1),
    ["J vltasl rlhr ", "zdfog odxr ypw", " atasl rlhr p ", "gwkzzyq zntyhv", " lvz wp!!!"])

assert_equals(
    demoving_shift(["J vltasl rlhr ", "zdfog odxr ypw", " atasl rlhr p ", "gwkzzyq zntyhv", " lvz wp!!!"], 1),
    "I should have known that you would have a perfect answer for me!!!")


# Other's Answer
# Zelouille, kickh, rondim

from string import ascii_lowercase as abc, ascii_uppercase as ABC
from math import ceil


def _code(string, shift, mode):
    return ''.join(
        abc[(abc.index(c) + i*mode + shift) % len(abc)] if c in abc else
        ABC[(ABC.index(c) + i*mode + shift) % len(ABC)] if c in ABC else c
        for i, c in enumerate(string))


def moving_shift(string, shift):
    encoded = _code(string, shift, 1)
    cut = int(ceil(len(encoded) / 5.0))
    return [encoded[i: i+cut] for i in range(0, 5 * cut, cut)]


def demoving_shift(arr, shift):
    return _code(''.join(arr), -shift, -1)


# lechevalier, berylluolan

def caesar_cipher(mode, s, shift):
    return ''.join(chr((ord(c) - 65 - 32 * c.islower() + i * mode) % 26
                       + 65 + 32 * c.islower()) if c.isalpha() else c for i, c in enumerate(s, shift))


def moving_shift(s, shift):
    ret = caesar_cipher(1, s, shift)
    return [ret[i: i - -len(ret) / 5] for i in range(0, len(ret) + bool(len(ret) % 5), -(-len(ret) / 5))]


def demoving_shift(s, shift):
    return caesar_cipher(-1, ''.join(s), shift)
