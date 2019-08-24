"""
Instructions
The Fibonacci numbers are the numbers in the following integer sequence (Fn):
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, ...
such as
    F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.

Given a number, say prod (for product), we search two Fibonacci numbers F(n) and F(n+1) verifying
    F(n) * F(n+1) = prod.

Your function productFib takes an integer (prod) and returns an array:
[F(n), F(n+1), true] or {F(n), F(n+1), 1} or (F(n), F(n+1), True)
depending on the language if F(n) * F(n+1) = prod.
If you don't find two consecutive F(m) verifying F(m) * F(m+1) = prodyou will return
[F(m), F(m+1), false] or {F(n), F(n+1), 0} or (F(n), F(n+1), False)
F(m) being the smallest one such as F(m) * F(m+1) > prod.
Examples
productFib(714) # should return [21, 34, true],
                # since F(8) = 21, F(9) = 34 and 714 = 21 * 34
productFib(800) # should return [34, 55, false],
                # since F(8) = 21, F(9) = 34, F(10) = 55 and 21 * 34 < 800 < 34 * 55
Notes: Not useful here but we can tell how to choose the number n up to which to go: we can use the "golden ratio" phi which is (1 + sqrt(5))/2 knowing that F(n) is asymptotic to: phi^n / sqrt(5). That gives a possible upper bound to n.
You can see examples in "Example test".
References
http://en.wikipedia.org/wiki/Fibonacci_number
http://oeis.org/A000045
Fundamentals
"""


def F(n):
    if n == 0:
        result = 0
    elif n == 1:
        result = 1
    elif n > 1:
        result = F(n - 1) + F(n - 2)
    return result


def productFib(prod):
    Fib_list = [0, 1]
    tmp = Fib_list[-1] + Fib_list[-2]
    while tmp <= prod:
        Fib_list.append(tmp)
        tmp = Fib_list[-1] + Fib_list[-2]

    #     print(Fib_list)
    #     tmp2 = Fib_list[i]*Fib_list[i+1]
    for i in range(len(Fib_list)):
        #         print(i, Fib_list[i])
        if Fib_list[i] * Fib_list[i + 1] < prod:
            pass
        elif Fib_list[i] * Fib_list[i + 1] == prod:
            return [Fib_list[i], Fib_list[i + 1], True]
        else:
            return [Fib_list[i], Fib_list[i + 1], False]



# Other's solutions
# student003, kevinhwang91, nvilela01, rasmus_ri, mak600d, jesonnn (plus 44 more warriors)
def productFib(prod):
    a, b = 0, 1
    while prod > a * b:
        a, b = b, a + b
    return [a, b, prod == a * b]

# evanxq, Lydiawu, rohin993, batTrung
def productFib(prod):
    a,b = 0,1
    while a*b < prod:
        a,b = b, b+a
    return [a, b, a*b == prod]

# anvatar, quneha, BondarenkoEkat
def productFib(prod):
    func = lambda a, b: func(b, a+b) if a*b < prod else [a, b, a*b == prod]
    return func(0, 1)

