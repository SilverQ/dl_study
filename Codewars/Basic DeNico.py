"""
Basic DeNico
Instructions
Task
Write a function deNico/de_nico() that accepts two parameters:
key/$key - string consists of unique letters and digits
message/$message - string with encoded message
and decodes the message using the key.

First create a numeric key basing on the provided key by assigning each letter position
in which it is located after setting the letters from key in an alphabetical order.
For example, for the key crazy we will get 23154 because of acryz (sorted letters from the key).
Let's decode cseerntiofarmit on using our crazy key.

1 2 3 4 5
---------
c s e e r
n t i o f
a r m i t
o n

After using the key:

2 3 1 5 4
---------
s e c r e
t i n f o
r m a t i
o n

Notes
The message is never shorter than the key.
Don't forget to remove trailing whitespace after decoding the message
Examples

deNico("crazy", "cseerntiofarmit on  ") => "secretinformation"
deNico("abc", "abcd") => "abcd"
deNico("ba", "2143658709") => "1234567890"
deNico("key", "eky") => "key"

Check the test cases for more examples.
Related Kata

Basic Nico - encode

"""

import numpy as np
import math


def make_key(str):
    key_index = {c: i+1 for i, c in enumerate(sorted(set([c for c in str])))}
    key = [key_index[c] for c in str]
    return key


def de_nico(key, msg):
    key_index = make_key(key)
    msg_split = [msg[i:i + len(key)] for i in range(0, len(msg), len(key))]
    result_str = ''.join(''.join([tmp[int(k)-1] for k in key_index if int(k) <= len(tmp)]) for tmp in msg_split)
    return result_str.rstrip()


def de_nico_old(key, msg):
    key_index = make_key(key)
    print(key, len(key_index), key_index)  # 23154
    msg_split = [msg[i:i + len(key)] for i in range(0, len(msg), len(key))]
    print(msg_split)
    # for tmp in msg_split:
    #     print(tmp, len(tmp))
    #     for i in key_index:
    #         if int(i) <= len(tmp):
    #             print(i, tmp[int(i)-1])
    #     tmp1 = ''.join([tmp[int(k)-1] for k in key_index if int(k) <= len(tmp)])
    #     print(tmp, '->', tmp1)
    result_str = ''.join(''.join([tmp[int(k)-1] for k in key_index if int(k) <= len(tmp)]) for tmp in msg_split)
    # print(result_str)
    return result_str.rstrip()


# Test.describe("Basic tests")
# Test.assert_equals(de_nico("crazy", "cseerntiofarmit on  "), "secretinformation")
# Test.assert_equals(de_nico("crazy", "cseerntiofarmit on"), "secretinformation")
# Test.assert_equals(de_nico("abc", "abcd"), "abcd")
# Test.assert_equals(de_nico("ba", "2143658709"), "1234567890")
# Test.assert_equals(de_nico("a", "message"), "message")
# Test.assert_equals(de_nico("key", "eky"), "key")


# de_nico("crazy", "cseerntiofarmit on  ")
# de_nico("crazy", "cseerntiofarmit on")
de_nico('ifjlkopsdy', 'wookmgsjizupslimptmwd uhwqr')    # 'ookgmsjiwzsplmiptmuwu hqwrd'
# de_nico('mljkcrpsqh', 'ylryipucrvxgwixd')               # 'piryyruvcldxwixg'
