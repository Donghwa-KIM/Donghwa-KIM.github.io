---
layout: post
comments: true
title: Decorators
categories: Advanced Python
tags:
- Advanced Python
---

**<span style='color:DarkRed'>Decorators: reference</span>**

- 이 글에서 작성된 Decorator 개념은 [CNN visualization]({{ site.baseurl }}/gradients_based.html)에 활용 된다.
- Decorators(@function)란 해당 함수에 reference를 다는 것이다.
- 일반적으로 decorator는 특정 함수를 입력으로 받는다는 점에서 함수 안에 함수의 구조를 가진다.
- 먼저 아래의 일반적인 python 함수 구조를 살펴본 뒤 decorator 예제를 설명하고자 한다.
	- Case 1: 함수안에 함수
	- Case 2: 함수안에 함수를 인자로 입력
		- $f(\text{함수})$
	- Case 3: 함수가 함수를 return하는 경우

- 전체 코드는 <a href="https://github.com/Donghwa-KIM/grad_CAM/blob/master/python_advance.ipynb">여기</a>에 참조되어 있다.

```python
def f():
    
    def g():
        print("Hi, it's me 'g'") # 3
        print("Thanks for calling me") # 4
        
    print("This is the function 'f'") # 1
    print("I am calling 'g' now:") # 2
    g()

    
f()
```

    This is the function 'f'
    I am calling 'g' now:
    Hi, it's me 'g'
    Thanks for calling me



```python
succ(10)
```




    11


<br>


- succ 함수와 successor 함수는 같은 함수임


```python
successor == succ
```




    True

<br>

- succ 함수를 지워져도 successor 함수는 지워지지 않음


```python
del succ
```

<br>

---

- Case 1 : 함수안에 함수

실행순서: $f() -> g()$


```python
successor
```




    <function __main__.succ(x)>




```python
def temperature(t):
    def celsius2fahrenheit(x):
        return 9 * x / 5 + 32

    result = "It's " + str(celsius2fahrenheit(t)) + " degrees!" 
    return result

print(temperature(20))
```

    It's 68.0 degrees!

<br>

---


- Case 2: 함수안에 함수를 인자로 입력
    - f(함수)


```python
def g():
    print("Hi, it's me 'g'")
    print("Thanks for calling me")
    
def f(func):
    print("Hi, it's me 'f'")
    print("I will call 'func' now")
    func()
          
f(g)
```

    Hi, it's me 'f'
    I will call 'func' now
    Hi, it's me 'g'
    Thanks for calling me

<br>


```python
import math

def foo(func):
    print("The function " + func.__name__ + " was passed to foo")
    res = 0
    for x in [1, 2, 2.5]:
        res += func(x)
    return res

print(foo(math.sin))
print(foo(math.cos))
```

    The function sin was passed to foo
    2.3492405557375347
    The function cos was passed to foo
    -0.6769881462259364

<br>

---

- Case 3: 함수가 함수를 return하는 경우


```python
def f(x):
    def g(y):
        return y + x + 3 
    return g

nf1 = f(1)
nf2 = f(3)

print(nf1)
print(nf2)
```

    <function f.<locals>.g at 0x10478d268>
    <function f.<locals>.g at 0x1044708c8>



```python
print(nf1(1))
print(nf2(1))
```

    5
    7

<br>

---
**<span style='color:DarkRed'>Decorator 예제</span>**


```python
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return function_wrapper

def foo(x):
    print("Hi, foo has been called with " + str(x))
    return str(x)

print("We call foo before decoration:")
foo("Hi")
    
print("We now decorate foo with f:")
foo = our_decorator(foo)

print("We call foo after decoration:")
foo(42)
```

    We call foo before decoration:
    Hi, foo has been called with Hi
    We now decorate foo with f:
    We call foo after decoration:
    Before calling foo
    Hi, foo has been called with 42
    After calling foo

<br>

- decorator는 함수를 입력으로 받아 함수를 반환한다.
- 아래의 예제에서 ```foo```이라는 함수는 
```@our_decorator```함수의 입력을 받아 새로운(꾸며진) 함수 ```foo```를 반환하게 된다.

```python
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return function_wrapper

@our_decorator 
def foo(x):
    print("Hi, foo has been called with " + str(x))
foo("Hi")
```

    Before calling foo
    Hi, foo has been called with Hi
    After calling foo

<br>

```python
def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        res = func(x)
        print(res)
        print("After calling " + func.__name__)
    return function_wrapper

@our_decorator
def succ(n):
    return n + 1

succ(10)
```

    Before calling succ
    11
    After calling succ

<br>

```python
from math import sin

def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        res = func(x)
        print(res)
        print("After calling " + func.__name__)
    return function_wrapper

sin = our_decorator(sin)

```


```python
sin(3.1415)
```

    Before calling sin
    9.265358966049024e-05
    After calling sin

<br>

```python
def argument_test_natural_number(f):
    def helper(x):
        if type(x) == int and x > 0:
            return f(x)
        else:
            raise Exception("Argument is not an integer")
    return helper
    
@argument_test_natural_number
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)



factorial(3)
```




    6

<br>

- Call로 value 저장하는 방법





```python
def call_counter(func):
    def helper(x):
        helper.calls += 1
        return func(x)
    helper.calls = 0

    return helper

@call_counter
def succ(x):
    return x + 1

print(succ.calls)
for i in range(10):
    succ(i)
    
print(succ.calls)
```

    0
    10


<br>

- ```class``` 활용한 decorate
	- 함수 대신 클래스
- ```__call__```: instance를 다시 함수로 선언해주는 역할을 해준다


```python
class A:
    
    def __init__(self):
        print("An instance of A was initialized")
    
    def __call__(self, *args, **kwargs):
        print("Arguments are:", kwargs,args )
              
x = A()
print("now calling the instance:")
x(3, 4, 5 , x=11, y=10)
print("Let's call it again:")
x(3, 4, x=11, y=10)
```

    An instance of A was initialized
    now calling the instance:
    Arguments are: {'x': 11, 'y': 10} (3, 4, 5)
    Let's call it again:
    Arguments are: {'x': 11, 'y': 10} (3, 4)


---


```python
class Fibonacci:

    def __init__(self):
        self.cache = {}
tmp = Fibonacci()
tmp.cache
```




    {}


<br>


```python
def decorator1(f):
    def helper():
        print("Decorating", f.__name__)
        f()
    return helper

@decorator1
def foo():
    print("inside foo()")

foo()
```

    Decorating foo
    inside foo()

<br>


```python
class decorator2:
    
    def __init__(self, f):
        self.f = f
        
    def __call__(self):
        print("Decorating", self.f.__name__)
        self.f() # "inside foo()"

@decorator2
def foo():
    print("inside foo()")

foo()
```

    Decorating foo
    inside foo()

<br>

---

**<span style='color:DarkRed'>Reference</span>** 


https://www.python-course.eu/python3_decorators.php
