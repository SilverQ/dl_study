# Python3 Programming(위키북스)
# 05. 클래스(85 page)

"""
클래스(Class) 개념
    객체를 만들어 내기 위한 설계도 혹은 틀, 연관되어 있는 변수와 메서드의 집합
https://gmlwjd9405.github.io/2018/09/17/class-object-instance.html

클래스를 정의하면 일종의 도장 역할을 하며, 도장을 찍듯이 인스턴스 객체를 생성하여 사용함.
기본적으로 인스턴스 객체는 생성 직후, 원본 클래스와 동일한 데이터와 함수를 가짐
객체 vs 인스턴스 : 객체는 구현할 대상, 클래스에 선언한 모양 그대로 생성된 실체이고, 객체가 메모리에 할당되어 실제 사용될 때 인스턴스
"""


# 클래스는 통상 데이터나 메서드로 이루어지지만, 필수적으로 있어야 하는 것은 아님.
# 간단한 경우의 클래스 선언을 통해 새로운 이름공간 생성을 해보자.
class MyClass:
    pass


print(type(MyClass))    # <class 'type'>


# 변수와 메서드를 가진 클래스를 정의해보자.
class Person:
    Name = 'Default Name'

    def print_name(self):
        print('My name is {}'.format(self.Name))


p1 = Person()
p1.print_name()      # My name is Default Name

p2 = Person()
p2.Name = 'HDH'
p2.print_name()      # My name is HDH


# 변수나 함수의 이름 탐색 순서 : 인스턴스 객체 영역 -> 클래스 객체 영역 -> 전역 영역

print("p1's name: ", p1.Name)   # p1's name:  Default Name
print("p2's name: ", p2.Name)   # p2's name:  HDH
# 앞서 p2.Name = 'HDH' 명령을 통해 인스턴스 p2의 이름공간에 있는 Name 변수를 바꿨었다.

p2.Age = '42'
print(p2.Age)
# print(p1.Age)       # AttributeError: 'Person' object has no attribute 'Age'

Person.title = 'New Title'
p2.title = 'My Title'
print("p1's title: ", p1.title)     # Class에 title 변수를 추가했더니, p1이 Class의 title 변수를 공유하고 있다.
print("p2's title: ", p2.title)

