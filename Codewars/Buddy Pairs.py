"""
Buddy pairs
You know what divisors of a number are. The divisors of a positive integer n are said to be proper when you consider only the divisors other than n itself. In the following description, divisors will mean proper divisors. For example for 100 they are 1, 2, 4, 5, 10, 20, 25, and 50.
Let s(n) be the sum of these proper divisors of n. Call buddy two positive integers such that the sum of the proper divisors of each number is one more than the other number:
(n, m) are a pair of buddy if s(m) = n + 1 and s(n) = m + 1

For example 48 & 75 is such a pair:
    Divisors of 48 are: 1, 2, 3, 4, 6, 8, 12, 16, 24 --> sum: 76 = 75 + 1
    Divisors of 75 are: 1, 3, 5, 15, 25 --> sum: 49 = 48 + 1
Task
Given two positive integers start and limit, the function buddy(start, limit) should return the first pair (n m) of buddy pairs such that n (positive integer) is between start (inclusive) and limit (inclusive); m can be greater than limit and has to be greater than n
If there is no buddy pair satisfying the conditions, then return "Nothing" or (for Go lang) nil
Examples
(depending on the languages)
buddy(10, 50) returns [48, 75]
buddy(48, 50) returns [48, 75]
or
buddy(10, 50) returns "(48 75)"
buddy(48, 50) returns "(48 75)"
Note
    for C: The returned string will be free'd.
    See more examples in "Sample Tests:" of your language.
"""


def s_list(n):
    return [i for i in range(1, int(n/2)+1) if n % i == 0]


def s_old(n):
    return sum([i for i in range(1, int(n/2)+1) if n % i == 0]) - 1


def s(n):
    sum_s = 0
    i = 2
    while i <= int(n/i):
        if n % i == 0:
            sum_s += i + (0 if i == int(n/i) else int(n/i))
        i += 1
    return sum_s


def buddy(start, limit):
    results = 'Nothing'
    for i in range(start, limit):
        if i % 2 == 0:
            tmp1 = s(i)
            # print(i, tmp1)
            if tmp1 >= start:
                tmp2 = s(tmp1)
                # print(i, tmp1, tmp2, tmp2 == i)
                if tmp2 == i:
                    results = [i, tmp1]
                    # print(results)
                    break
    return results
# 58710에서 STDERR 에러 발생


# print(buddy(57345, 90061))
# Test.assert_equals(buddy(10, 50), [48, 75])
# Test.assert_equals(buddy(2177, 4357), "Nothing")
# Test.assert_equals(buddy(57345, 90061), [62744, 75495])
# Test.assert_equals(buddy(1071625, 1103735), [1081184, 1331967])
# print(buddy(10, 50))
# Test.assert_equals(buddy(1071625, 1103735), [1081184, 1331967])
# print(buddy(10, 50))
# print(buddy(2177, 4357))
# print(buddy(57345, 90061))
# print(buddy(1071625, 1103735))

print(s_list(1081184))
print(s_list(1331967))
# [1, 2, 4, 8, 13, 16, 23, 26, 32, 46, 52, 92, 104, 113, 184, 208, 226, 299, 368, 416, 452, 598, 736, 904, 1196, 1469, 1808, 2392, 2599, 2938, 3616, 4784, 5198, 5876, 9568, 10396, 11752, 20792, 23504, 33787, 41584, 47008, 67574, 83168, 135148, 270296, 540592]
# [1, 3, 7, 13, 17, 21, 39, 41, 49, 51, 91, 119, 123, 147, 221, 273, 287, 357, 533, 637, 663, 697, 833, 861, 1547, 1599, 1911, 2009, 2091, 2499, 3731, 4641, 4879, 6027, 9061, 10829, 11193, 14637, 26117, 27183, 32487, 34153, 63427, 78351, 102459, 190281, 443989]

print(len(s_list(1081184)))     # 47
print(len(s_list(1331967)))     # 47

print(s_list(62744))     # 31
print(s_list(75495))     # 15
# [1, 2, 4, 8, 11, 22, 23, 31, 44, 46, 62, 88, 92, 124, 184, 248, 253, 341, 506, 682, 713, 1012, 1364, 1426, 2024, 2728, 2852, 5704, 7843, 15686, 31372]
# [1, 3, 5, 7, 15, 21, 35, 105, 719, 2157, 3595, 5033, 10785, 15099, 25165]

print(len(s_list(62744)))     # 31
print(len(s_list(75495)))     # 15

# Run Sample Tests까지는 통과(5203ms소요)했으나, Attempt까지는 실패(Execution Timed Out)함
# 보다 빠르게 해결할 수 있는 방안을 찾아보쟈.
