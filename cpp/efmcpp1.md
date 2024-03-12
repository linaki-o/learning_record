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
```

# 基础17: 返回值优化

**一个很有意思的现象就是你局部变量移动一下居然就不被回收了???**


具体实践可以会看课程

返回值优化的case:
- 未具名返回值优化：函数返回一个未具名对象时有两次构造 优化方法其实就是改成传引用 C++14后默认开启 C++17开始强制开启
```cpp
void optimize(Test &test) {
    new (&test) Test(1);
    ... # other operation
}
```
- 返回一个具名对象：就是有变量名的对象 两种优化 一种具名(两次拷贝全优化了, 编译option可以禁止) 一种不具名（优化掉了第二次拷贝）

返回值优化失效的case(NRVO 具名对象优化失效):
- 可能返回不同对象 主要原因是当需要对对象做额外处理时不方便
- 返回全局变量 因为生命周期很长 所以不能move
- 返回函数参数引用
- 存在赋值行为 主要原因就是赋值函数是必须的不能去
```cpp
Test fun1() {
    return Test(10);
}
int main() {
    Test result(20);
    result = fun1();
}
```
- 返回成员变量
- 使用std::move返回

# 条款25：对右值引用使用std::move对通用引用使用std::forward
const引用与右值引用重载来提高效率
- 仍不够高效
- 参数多了重载很麻烦

使用通用引用可以更好的完成任务

引用的类型与被引用的类型应该完全一样
```cpp
w.setName("LK"); // 虽然是左值但 调用的是右值引用 
/*
underlying operations
std::string temp("LK");
std::string name = temp;
*/
```

返回通用引用要用forward
返回右值引用要用move
```cpp
Widget func(Widget &&w) { // 右值引用传进来就说明这个变量不要了 随便你处理
    return std::move(w); // 不用move会拷贝
}
template <typename T>
auto&& func(T&& w) { 
    return std::forward<T>(w); // 不能用move，因为如果是左值的话不能把它move掉了 
}
```

# 基础18：emplace_back与push_back

emplace_back可以只做一次构造

emplace_back是万能引用

emplace_back的编译期性能会差一点 因为会推导成不同的类型(对于字符数组来说)

# 条款26：避免在通用引用上做重载
编译器首选最匹配 最高效的函数

类型不同
```cpp
// logAndAdd做了一次int参数类型的重载
short nameIdx;
logAndAdd(nameIdx); // 走的却是通用引用 因为short to int要转换 
```
即使只是常量性不同也会选择通用引用的函数

# 基础19：模板元编程初探，SFINAE，enable_if

元编程就是编程的编程，在一种编程语言的基础上的编程语言

模板元编程：使用模板编写程序的程序 编译期执行的
模板元函数：编译器执行且输入输出可为数值，也可以类型
```cpp
template <typename T>
struct Func1 {
    using type = T;
};
template <int n> // 这里的模板值只能为bool类型或整型
struct Func2 {
    int data[n];
};
```

SFINAE(替换失败并非错误) 就是拿一种模板的错误来减弱通用引用的适配性 以此来解决条款26的问题
```cpp
class Person {
public:
    template <typename T, typename = typename Func1<T>::type_b> // 模板推导失败就不走这了 有个疑问 那你写这个函数的意义是什么 别写了呗
    Person(T &&n):... {
        ...
    }
};
```

enable_if(用来解决上面的疑问的)
```cpp
template<bool, typename _Tp = void>
struct enable_if {};
template<typename _Tp>
struct enable_if<true, _Tp> {
    typedef _Tp type;
};
```

# 条款41：对于移动成本低且总是被拷贝的可拷贝形参，建议按值传递

```cpp
class Widget3 {
    void addTest(Test newTest) {
        _tests.push_back(std::move(newTest));
    }
};
// 传左值 1copy 1move
// 传右值 2move(返回值优化可以优化掉一次move)
```
与左/右引用重载相比: 维护的少了
与万能引用相比: 比较简单 可预测

通过拷贝赋值与移动赋值，按值传递时要多考虑
```cpp
class Password {
    void changeTo(string newPwd) { // 一次拷贝构造 一次移动赋值 一次申请内存 一次释放内存
        text = std::move(newPwd);
    }
    void changeTo(const string& newPwd) { // 一次拷贝赋值 当text的长度大于newPwd时直接拷贝就好 就不需要申请了
        text = newPwd;
    }
}
```

# 条款27：熟悉通用引用重载的替代方法

一：按值传递

二：
```cpp
template <typename T>
void logAndAddImpl(T &&name, std::false_type) {
    names.emplace_back(std::forward<T>(name));
}
std::string nameFromIdx(int idx);
void logAndAddImpl(int idx, std::true_type) {
    names.emplace_back(nameFromIdx(idx));
}
template <typename T>
void logAndAdd(T &&name) {
    logAndAddImpl(std::forward<T>(name), std::is_integral<typename std::remove_reference<T>::type>());
}
```

三：enable_if
```cpp
template <
    typename T,
    typename = std::enable_if_t<
    !std::is_base_of<Base, std::decay_t<T>>::value &&
    !std::is_integral<std::remove_reference_t<T>>:value>
    >
explicit Base(T&& n) {
    ...
}
```

c++11引入char16_t类型
而这个类型与char不同，是不能隐式转换为string的
而且出错了很难排查, 再加一个判断
```cpp
static_assert(
    std::is_constructible<std::string, T>::value,
    "..."
);
```

# 条款28：理解引用折叠

左值引用+左值引用依然是左值引用 只要有一个左值引用那么就是左值引用

发生的四种情况：
- 模板实例化
- auto类型推导
- typedef与别名声明
- decltype

# 条款29：假定移动操作不存在，成本高，未被使用


std::array的移动操作
array是在栈上分配的，移动操作是线性的，与vector相比还是逊色很多

std::string的small string optimization（SSO）：小字符串存储在string的缓冲区即栈区中

移动操作没有声明noexcept时不敢用

# extern与static

c/c++都遵循单定义原则

对于函数来说，声明和定义是不同的

对变量来说就有点含糊了, 所以用extern表明变量声明

用static来限定作用域：
- 修饰普通函数
- 全局变量
- 局部变量
- 类成员函数
- 类成员属性：只能类内声明，类外定义
- const static 类成员属性：可在类内声明时赋值，但依然是声明

# 条款30：完美转发失败的情况

花括号初始化器：模板函数不能推导{}

0或NULL作为空指针：因为会被推导为int类型, 而不是指针类型

仅有声明的static const数据成员：万能引用需要真真切切存在的对象，而声明并没有创建对象，没有内存, 而普通形参可以用static const是因为编译器会对此类成员实行常量传播

重载函数名称与模板名称：函数居然也是可以取地址的，因为不知道该如何推导。用using与类型转换限定一下类型就好了

位域：不能取地址 地址是以字节为单位的 而位域是以bit为单位的

# 条款31：避免使用默认捕获模式

默认按引用捕获局部变量要考虑其生命周期

默认按值捕获类内属性，要考虑this指针引发的问题
```cpp
class Widget {
public:
    void addFilter() {
        filter.emplace_back([=](int val){...divisor...}); // 捕获不到divisor 捕获到的是this指针
    }
private:
    int divisor = 0;
}
```

默认按值捕获，要考虑其对局部static变量的依赖
```cpp
vector<function<void(int)>> filter;
void addDivisorFilter() {
	static int divisor = 10;
	divisor++;
	filter.emplace_back([=](int val) { // 闭包了 用divisor=divisor就是按值捕获
		cout << divisor << endl;
	});
}
int main() {
	addDivisorFilter();
	addDivisorFilter();
	addDivisorFilter();
	addDivisorFilter();
	addDivisorFilter();
	addDivisorFilter();
	auto func = *filter.begin();
	func(0); // 16
}
```

在C++中，lambda表达式可以捕获静态局部变量。静态局部变量在函数内部声明，但其生命周期与程序的运行时间相同，并且只会初始化一次。Lambda表达式可以捕获静态局部变量，这意味着它们可以在闭包内使用这些变量的值。

# 基础21：std::bind初探

绑定成员函数
```cpp
auto wadd = std::bind(&Widget::add, &w, 10, std::placeholders::_2);
```

```cpp
void fun1(Test &t);
Test t(50);
auto funBind1 = std::bind(fun1, std::ref(t)); // 这样才是真正的传引用 否则会调用拷贝构造
auto funBind1 = std::bind(fun1, t); // 左值拷贝构造
auto funBind1 = std::bind(fun1, std::move(t)); // 右值移动构造
```

为了闭包所以不能用引用，bind会给你构造一份用

# 条款32；使用初始化捕获来移动对象到闭包中

或者创建一个重载了()的类

或者用std::bind

"对一个有左值引用参数的函数传递右值会发生什么"

# 条款33：对auto&&形参使用decltype以std::forward它们

```cpp
auto f = [](auto &&x) {
    return func(std::forward<decltype(x)>(x));
};

// 当传右值时会出现T&&&&的情况
template <typename T>
T&& forward(remove_reference_t<T>& param) {
    return static_cast<T&&>(param);
}
```

"模板与auto推导"

# 条款34：考虑lambda而非std::bind

1. 当函数作为std::bind参数时会立刻执行
```cpp
auto func = std::bind(func1, func2(), _1); // func2() 会立刻执行

auto func = std::bind(func1, std::bind(func2), _1); // func2() 会在调用func时执行
```

2. 当存在函数重载时bind会存在问题
```cpp
void func(int a, int b, int c);
void func(int a, int b);
auto funcA = std::bind(func, 5, std::placeholders::_1, 10); // 傻傻分不清是哪一个func 报错 解决方法：用using加static_cost限定一下函数的类别
// 也会有一些性能问题，因为用std::bind时用不了内联这个功能
```

3. 功能稍微复杂的情况。看视频，有点抽象

4. 合理使用std::bind的情况
- c++11不提供移动捕获，可使用lambda与bind相结合
- c++11中bind对象上的函数调用运算符使用完美转发，可接受任意参数，lambda不行，当时lambda还不接受auto类型参数呢