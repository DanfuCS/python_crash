def func(*args, **kwargs):
    print(f"Hello, {args[0]}")  # *args：当传入的参数个数未知，且不需要知道参数名称时
    for arg in args:
        print(f"arg: {arg}")

    print(kwargs)  # **kwargs：允许将不定长度的键值对作为参数传递给一个函数


func("danfu", 3, 5, 7, 9, x=0, y=10, z=100)
