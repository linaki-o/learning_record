# 基础13：堆栈内存分配流程与内存泄漏
我发现我真是指针高手 他说的我都会
```cpp
int c = 30;
func(&(&c)); // 不得行 因为&c是右值
```
# 基础14：申请内存的各种方法

new/delete的底层实现
```cpp
// 构造函数
void* mem = operator new(sizeof(Test));
Test* p2 = (Test*) mem;
p2->Test::Test();
// 析构
p2->~Test();
operator delete(p2);
```

placement new
```cpp
Test* p3 = new(p1)Test(); // 需要传入分配好内存的指针 其本身不会额外申请空间

void* mem = operator new(sizeof(Test), p1);
```

可以重写
```cpp
void *operator new(size_t size);
void *operator new(size_t size, void* buf);

void operator delete(void* ptr, size_t size); // 这个size可加可不加
```

# 基础15：重写operator new/delete的意义

首先当你new对象时并不是仅仅给你分配对象的大小，还会需要一些空间来存储其他信息

其次malloc这个动作是有代价的，希望次数越少越好

所以在一些需要重复使用new的场景下就有重写的必要

示例代码中用union节省空间的操作很聪明

# 基础16：Array new，Array delete与std::allocator的引入

基础用法
```cpp
int *a = new int[3]{1, 5, 23};
delete[] a; // 这个[]加不加都行，因为本身就给你记录了大小
```

delete[]的作用主要是会给你调用多次析构

若每一个类都要重载一遍，导致代码重复，非常麻烦，且修改成本高，这时std::allocator应运而生

# 条款18，19：智能指针

智能指针性能不差原始指针，且更安全

用模板实现可变参数

```cpp
std::unique_ptr<Investment, void(*)(Investment*)> uptr2(nullptr, delfunc2); // size为16byte
std::unique_ptr<Investment, decltype(delfunc1)> uptr2(nullptr, delfunc1); // size为8byte
```

要是lambda表达式中具有很多状态的话，产生的std::unique_ptr也会很大

# 条款24：区分通用引用与右值引用

通用引用的两个条件：
- T&&
- 类型推导
```cpp
template<class T, class Allocator=allocator<T>>
class vector {
public:
    push_back(T&& x); // 这边是右值引用 因为没有满足类型推导的条件
}
```

可变参数模板的通用引用要求：
- Args&&...
- 类型推导
```cpp
template<typename T>
class vector {
public:
    template<typename... Args>
    void emplace_back(Args&&... args);
}
```

auto&&与auto&&... 要求与上同
```cpp
auto timeFuncInvocation = [](auto&& func, auto&&... params) {
    std::forward<decltype(func)>(func)(
        std::forward<decltype(params)>(params)...);
        ...
};