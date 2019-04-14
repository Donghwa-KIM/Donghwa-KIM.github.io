---
layout: post
comments: true
title: Classmethod & Staticmethod with Decorators
categories: Advanced Python
tags:
- Advanced Python
---






- ```classmethod()```: 중복되는 코드를 줄이기 위해, ```class```안에 있는 ```function```을 재귀적인 ```class```(```classmethod```)의 형태로 표현할 때 사용될 수 있다.

- ```staticmethod()```: ```classmethod()```와 유사하지만, ```class```와 독립적으로 개별적인 함수로 표현할 수 있다.

---

**<span style='color:DarkRed'>classmethod</span>**
 

- 아래의 예제를 일반적으로 사용되는 class 문법으로 작성하면 몇가지 중복되는 내용이 있다.
    - ```__init__```함수과 ```display```의 내용이 ```fromBirthYear```에 포함되어 있다.


```python
from datetime import date

# random Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    # duplicated codes
    def fromBirthYear(self, name, birthYear):
        print(name + "'s age is: " + str(date.today().year - birthYear))
        
    def display(self):
        print(self.name + "'s age is: " + str(self.age))

person = Person('Adam', 19)
person.display()

person1 = person.fromBirthYear('John',  1985)
```

    Adam's age is: 19
    John's age is: 34

<br>

- 기존에 있는 ```class```를 활용해서 더 간결하게 표현할 수 이다.
- ```fromBirthYear```에 ```cls```인자를 받아 새로운 클래스를 할당하게 해보자.


```python
from datetime import date

# random Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    # another class
    def fromBirthYear(cls, name, birthYear):
        return cls(name, date.today().year - birthYear)

    def display(self):
        print(self.name + "'s age is: " + str(self.age))
```

<br>

- 아래 코드에서 ```fromBirthYear```는 클래스를 받는 함수이기 때문에, ```object(person.fromBirthYear)```가 아닌 ```classmethod(Person.fromBirthYear)```의 형태로 불러온다.
    - ```Person.fromBirthYear```는 class인것을 인지하자.
- 특별한 decorator(```@classmethod```)를 사용하기 않으면, 위 코드에 명시된 ```args(cls, name, birthYear)``` 3개로 인식되서 error를 반환한다.
    - ```@classmethod```: 특정함수의 전후로 ```classmethod```코드를 입히는 것이다.


```python
Person.fromBirthYear('John',  1985)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-135-52f9f151f2c4> in <module>
    ----> 1 Person.fromBirthYear('John',  1985)
    

    TypeError: fromBirthYear() missing 1 required positional argument: 'birthYear'


<br>

```python
from datetime import date

# random Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    @classmethod
    def fromBirthYear(cls, name, birthYear):
        return cls(name, date.today().year - birthYear)

    def display(self):
        print(self.name + "'s age is: " + str(self.age))

```


```python
Person.fromBirthYear('John',  1985).display()
```

    John's age is: 34

<br>

**<span style='color:DarkRed'>staticmethod</span>**
 


```python
from datetime import date

# random Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @staticmethod
    def printName(name,birthYear):
        print(name + "'s birthYear is: " + str(birthYear))
    
    @staticmethod
    def fromBirthYear(name, birthYear):
        return Person(name, date.today().year - birthYear)

    def display(self):
        print(self.name + "'s age is: " + str(self.age))

```

<br>

- ```class(Person)```와 독립적으로 일반적인 함수로 사용할수도 있다.


```python
Person.printName('John',  1985)
```

    John's birthYear is: 1985

<br>

- ```cls``` 대신에 새롭게 클래스(e.g.```Person()```)를 정의해 ```classmethod```와 동일한 목적으로 사용될 수 있다.


```python
Person.fromBirthYear('John',  1985).display()
```

    John's age is: 34


**<span style='color:DarkRed'>Reference</span>**

- https://www.programiz.com/python-programming/methods/built-in/classmethod
- https://www.programiz.com/python-programming/methods/built-in/staticmethod
