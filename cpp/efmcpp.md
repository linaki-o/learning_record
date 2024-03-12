# 1
“If you’re teaching today what you were teaching five years ago, either the field is dead or you are”

应用领域：
- Operating systems
- Compilers
- Artificial Intelligence
- Image Editing
- Web browser
- High-Performance Computing
- Embedded systems
- Google and Microsoft use C++ for web indexing
- Scientific Computing
- Database
- Video Games
- Entertainment
- Finance


C++ Philosophy


Every second spent trying to understand the language is one not spent understanding the problem

The primary goal of the course is to drive who has previous experience with C/C++ and object-oriented programming to a proficiency level of (C++) programming

nullptr只能赋值给指针 nullptr is not a pointer, but an object of type nullptr_t

conversion rules:
- small integral type ⊗ small integral type → int
- signed type ⊗ unsigned type → unsigned type


Operation Ordering Undefined Behavior确实很奇怪，但平时并没有什么用处，有需要时再看

c++20 Spaceship Operator <=> 功能类似strcmp

C++20 introduces a set of functions <utility> to safely compare integers of different types (signed, unsigned)

# 11
A translation unit (or compilation unit) is the basic unit of compilation in C++. It consists of the content of a single source file, plus the content of any header file directly or indirectly included by it

不同变量会存储在程序的不同的段

# 简介
应用这些特性来写出正确、高效、可维护、可移植的程序

C++	所有版本
C++98	C++98和C++03
C++11	C++11和C++14
C++14	C++14

最遍布C++11各处的特性可能是移动语义: 右值对应于从函数返回的临时对象，而左值对应于你可以引用的（can refer to）对象，或者通过名字，或者通过指针或左值引用。

对于判断一个表达式是否是左值的一个有用的启发就是，看看能否取得它的地址。如果能取地址，那么通常就是左值。如果不能，则通常是右值。

这个启发的好处就是帮你记住，一个表达式的类型与它是左值还是右值无关。
```cpp
class Widget {
public:
    Widget(Widget&& rhs);   //rhs是个左值，
    …                       //尽管它有个右值引用的类型
};
```

右值副本通常由移动构造产生，左值副本通常由拷贝构造产生。

实参和形参的区别非常重要，因为形参是左值，而用来初始化形参的实参可能是左值或者右值。(有啥区别吗)

设计优良的函数是异常安全（exception safe）的，意味着他们至少提供基本的异常安全保证（即基本保证basic guarantee）。这样的函数保证调用者在异常抛出时，程序不变量保持完整（即没有数据结构是毁坏的），且没有资源泄漏。有强异常安全保证的函数确保调用者在异常产生时，程序保持在调用前的状态。

我定义一个函数的签名（signature）为它声明的一部分，这个声明指定了形参类型和返回类型。函数名和形参名不是签名的一部分。在上面的例子中，func的签名是bool(const Widget&)。

# 基础：顶层const与底层const
```cpp
// 顶层 修饰变量
const int a;
// 底层const 修饰指向的值
const int& a;
```
**当执行对象拷贝操作时，常量的顶层const不受影响，而底层const必须一致**
其实就是一些规则来时语言正常 不会出现歧义

引用不是对象且不进行拷贝
**const引用可以引用常量 可以引用任何东西**
非常量引用=常量 报错
引用如果作为右值，可以直接等价于其引用的对象
非常量=常量引用 正确的

According to the standard, in order to get reliable POSIX behavior, the first non-comment line of the Makefile must be **.POSIX**.


If you were to create the source files in the example and invoke make, you will find that it actually does know how to build the object files. This is because make is initially configured with certain inference rules, a topic which will be covered later. For now, we’ll add the **.SUFFIXES** special target to the top, erasing all the built-in inference rules.

Don’t use recursive Makefiles

在 GCC 编译器中，-MM 和 -MT 选项用于生成依赖关系信息，这些信息对于 make 工具来说非常重要，因为它们用于确定哪些源文件需要重新编译。
# 基础：值类型与右值引用
-fno-elide-constructors g++取消返回值优化的选项

左值：可以取地址的具体对象
右值：临时对象 不能取地址 **字符串除外的字面量**

不管你返回值是局部变量（确实该）还是全局变量 return时都是copy一份tmp对象
```cpp
int *p = &x++; // 报错
int *p = &++x;
```

赋值操作：
- 拷贝
- 引用
- 移动
```cpp
// 移动构造函数
Widget(Widget &&other) {
    ...
    mem = other.mem;
    other.mem = nullptr;
}
```

左值引用不接受右值
右值引用接受且仅接受右值
```cpp
int &x = -10; // 报错
int &&x = -10; // 这个x居然是可以改动的 即这个右值是可变的
```

用右值引用可以达到返回值优化的效果

std::move() 可以把一个左值变成右值 同时这个左值dead 失去了它的意义

将亡值就是用std::move()将泛左值转化为的右值
还可以通过static_cast<>()来转换

std::move()不移动

```cpp
#include <iostream>
#include <memory> // 引入智能指针头文件
#include <stdexcept> // 包含标准异常类

using namespace std;
class Widget {
public:
        Widget(){
                cout << "W ctor" << this << endl;
        }
        Widget(const Widget& w){
                cout << "W copy" << this << endl;
        }
        Widget(const Widget&& w){
                cout << "W move" << this << endl;
        }
        const Widget& operator=(const Widget&) {
                cout << "W copy assign" << this << endl;
                return *this;
        }
         const Widget& operator=(const Widget&&) {
                cout << "W move assign" << this << endl;
                return *this;
        }

        ~Widget(){
                cout << "W dtor" << this << endl;
        }

};
Widget getWidget() {
        Widget w = Widget();
        cout << "-------" << endl;
        return w;
}
int main() {
        Widget w;
        w = getWidget();
        for (int i = 0; i < 100; ++i) {
                cout << i << " ";
        }

        return 0;
}

/*
确实有temp对象，并且temp对象在完成使命后很快就析构了，不知道是什么原因
也有移动赋值函数
*/
```

右值引用是左值 只是别名
```cpp
Widget &&w = std::move(a);
```

右值 左值
指针 引用（就是别名）

当rhs是右值首先调用的就是移动

# 基础：数组指针与函数指针

数组指针与普通指针可不同
```cpp
int array[5] = {1,2,3,4,5}; // array的类型是int [5] 可不是int *
int *ptr = array; // 数组名退化为指针
int (*ptr2)[5] = &array; // 数组指针 []的优先级高于*
int (&ptr2)[5] = &array; // 数组引用 有点奇怪 为什么我是个引用还要取地址呢？
```

数组名作为参数名传递会退化
```cpp
// 三者等价
int fun(int a[100]);
int fun(int a[5]);
int fun(int* a);
// 不同于上者
int fun(int (*ptr1)[5]);
```

字符串字面量是左值 原因：字符串用寄存器构造太麻烦
```cpp
char str[2] = "h"; // "h"是const char[2]
const char *str = "h"; // 退化了
char *str = "h"; // 也行 但str[1] = 's';不行
int &ref[5]; // 会报错 因为没有引用数组

```

函数数据类型
```cpp
bool fun(int a, int b);
bool (*funptr)(int a, int b);
funptr = &fun; // &可以省略 相等于退化了
bool c = (*funpart)(1, 2); // (*)可以省略 函数指针使用


```

作为函数参数写一串很麻烦
```cpp
// 可以简化
typedef int int_name;
using int_name = int;
using fun3Ptr = bool (*)(int, int);
void func2(int c, fun3Ptr *fun);
```

函数指针作为返回值时不能用函数来替代 即(*)不可省略
```cpp
bool (*func3(int c))(int, int);
// 函数的引用也只能这么写
bool (&funref)(int, int) = fun; 
```

还是用别名方便
#  类型推导

模板中的const
```cpp
// 不管const放哪 T被推导成int * or not, const都表示顶层const
template<typename T>
void f(const T param);
void f(T const param);
```

```cpp
fun(int a);
fun(const int a);
// 不构成重载 即会冲突
```

```cpp
// 指针的引用
int *a = &b;
int *&c = a;
```

函数指针的底层const似乎只能用类型别名的方式表示出来
函数引用和函数指针的底层const会被编译器忽略 因为函数是在代码区，是只读不可写的

- 理解顶层与底层const
- 理解数组与函数会退化为指针
- 通用引用（万能引用）传入左值 ParamType为左值引用 传入右值 ParamType为右值引用

然后据此来推导
```cpp
template<typename T>
void f(ParamType param);

f(expr);
```

```cpp
template<typename T>
void f(T &param);

f(10); // 不行 T不能被推导成const int &

template<typename T>
void f(const T *param);
f(func); // 不行 函数指针的底层const似乎只能用类型别名的方式表示出来
```

```cpp
template<typename T>
void f(T &&param); // 通用引用 引用折叠：当T=int &时 ParamType=int &&&会折叠为int &
// 右值引用依然是左值

template<typename T>
void f(const T &&param); // 纯纯的右值引用 只能放右值
```

# 条款7：区别使用()和{}创建对象
c++11后

五种创建对象的方法
```cpp
// 下2等价
A a = 10; // 构造再拷贝 
A a = (10); // 构造再拷贝 
A a(10);
// 下2等价
A a{10};
A a = {10};

```

```cpp
A a = 10; // 只能接受一个参数
A a(10); // 作为参数与返回值时会进行拷贝出tmp对象
```
{} 完美解决问题 但作为返回值还是会拷贝 不允许参数缩窄转换

简化"聚合类"的初始化
```cpp
People a{10, "djs", 18.f};
```

聚合类：
- C++11
        - 所有成员都是public
        - 没有类内初始化
        - 没有定义任何构造函数
        - 没有基类，也没有virtual函数（C++17取消，可以有基类，但必须是公有继承，且必须是非虚继承，基类可以不是聚合类）


{}解决了C++的最令人头疼的解析问题
```cpp
void f(double value) {
        /*
                有两种解释
                1. 创建一个int参数
                2. 声明一个函数类型（被选中）
        */
        int i(int(value));
}
```

类内初始化不能用```int age(10);``` 依然是歧义的问题，会被认为是函数

奇怪的array
```cpp
S a[3] = {{1, 2}, {3, 4}, {5, 6}}; // ok
std::array<S, 3> a1 = {{1, 2}, {3, 4}, {5, 6}}; // 不行
std::array<S, 3> a1 = {{{1, 2}, {3, 4}, {5, 6}}}; // ok 因为array是有一个数组元素的聚合类
```


std::initializer_list<>可以使容器支持列表初始化
```cpp
std::vector<int> vec{1,2,3,4}; // 元素可以是任意个
```


当std::initializer_list<>作为参数来实现构造函数时，这个构造函数总是被优先匹配，甚至报错(在可能满足缩窄转换的情况下)

空的{}, 不会调用std::initializer_list构造，而是无参构造，但{{}} size=1， ({}) size=0

c++17 用之前的命令是关闭不了返回值优化的
编译器已经优化太多了
# 条款2：理解auto类型推导

auto和模板推导的方式很像，不同主要有以下方面：
- 万能引用的第二种写法 auto &&
- {}的类型推导
        ```cpp
        auto a1 = 27;
        auto a2(27);
        auto a3{27}; // int 所以auto a5{10, 27}; 是不行的
        auto a4 = {27}; // std::initializer_list
        ```
- template与std::initializer_list
        ```cpp
        template <typename T>
        void f(T param) {} 
        f({1,2,3}) // 报错 推导不出来
        template <typename T>
        void g(std::initializer_list<T> param) {} 
        g({1,2,3}) // ok
        ```
- auto做为返回值
        ```cpp
        // c++14中auto可作返回值, 但规则却是模板的规则
        auto f() {
                return {1, 2, 3}; 
        } 
        ```

# 条款9：优先考虑别名声明而非typedef

typename
```cpp
struct test1 {
        static int SubType; // 数据成员
};
struct test2 {
        typedef int SubType; // 类型
};
template <typename T>
class MyClass {
public:
        void foo() {
                /*
                当没加typename时，T::SubType默认被解释为数据成员    
                所以被解释为乘法 会报错
                加typename后即是类型
                */
                typename T::SubType *ptr;
        }
}
```

模板别名创建的两种方式
```cpp
template <class T>
using vector1 = std::vector<T>;

template <class T>
struct vector2 {
        typedef std::vector<T> type;
};
```

using比typedef的好处之一就是using不需要使用typename

而typedef如果不使用typename会发生歧义(都是作用域运算符的错)

**当编译器遇到类似T::xxx这样的代码时，它不知道xxx是一个类型成员还是一个数据成员，直到实例化时才能知道，但是为了处理模板，编译器必须知道名字是否表示一个类型，默认情况下C++假设通过作用域运算符访问的名字不是类型**

类型萃取器
```cpp
#include <type_traits>
std::remove_const<T>::type // c++11
std::remove_const_t<T> // c++14

template<class T>
using remove_const_t = typename remove_const<T>::type;
```

# 条款23：理解std::move和std::forward

```cpp
template <typename T>
T &mymove(T &&param) {
        return static_cast<T &>(param); // 当T为int &时依然会发生引用折叠即返回值为int &
}


// 自己搞一个std::move
template <typename T>
typename std::remove_reference<T>::type &&my_move(T &&param) {
        using ReturnType = typename std::remove_reference<T>::type &&; // c++11
        return static_cast<ReturnType>(param);
}
```

对一个对象使用std::move就是告诉编译器，这个对象很适合被移动，要注意这个很适合，即并不一定，特别是```const A &&```不能赋值给```A &&```, 而可以赋值给```const A&```，因为它谁都可以引用


std::forward<T>完美转发
```cpp
void process(const A& lval) {}
void process(A && rval) {}

template <typename T>
void logAndProcess(T &&param) {
        // 无论实参是左值还是右值，这时形参都是左值, 所以下面都是调用左值的那个
        // process(param);
        // 调用右值那个
        // process(std::move(param));
        
        process(std::forward<T>(param)); // 会根据实参类型选择
}
```

# 条款3：理解decltype类型推导

decltype + 变量
```cpp
int a[2] = {1, 2};
decltype(a) b; // b的类型为int [2] 所有信息都会被保留，不管是引用还是顶层const，数组与函数也不会退化
```

decltype + 表达式
```cpp
// 表达式不是左值就是右值 
// - 左值：得到该类型的左值引用
// - 右值：得到该类型
// 字面量走的都是表达式而非变量

int a = 10;
int *aptr = &a;
decltype(*aptr) b; // b的类型为int &
decltype((a)) b; // b的类型为int & too 想让一个变量作为表达式可以加括号

```

decltype并不会实际计算表达式的值，编译器会分析表达式并得到类型


decltype的使用场景
```cpp
/* 某些情况模板函数的返回值是无法提前确定的
如vector<bool> vec; vec[0]的返回值类型不是bool &

*/
// c++11
template <typename Container, typename Index>
auto testFun(Container &c, Index i)
        -> decltype(c[i])
{
        return c[i];
}


```

右值也可以放在左边，只要定义了赋值运算符, 这样其实就相当于调用一个函数了

```cpp
// c++14 可以让auto作为返回值 而auto作为返回值是用的模板的规则 所以返回值的类型不会是引用 所以会是个右值 故不能赋值 但自定义类型与bool类型还是有所不同
template <typename Container, typename Index>
auto testFun(Container &c, Index i)
{
        return c[i];
}
```

```cpp
// c++14 better的写法
// 可以使用decltype(auto)来保留xxx的所有修饰
template <typename Container, typename Index>
decltype(auto) testFun(Container &&c, Index i)
{
        // return c[i];
        return std::forward<Container>(c)[i];
}
```

# 基础4：Cpp类对象布局

影响cpp对象大小的3个因素：
- 非静态数据成员
- 虚函数
- 字节对齐

只要有虚函数就会在对象里有一个虚函数表指针 这个指针指向这个对象共用的虚函数表 这个表的第一个元素是type_info

字节对齐：
- 类中变量占有几个字节，那么这个变量应放在几的整数倍的位置上 long在win64上4字节 在linux64上8字节
- 结构体最后会向内部最大对象对齐
- 所以要注意类内属性的声明顺序

继承时的内存对齐
```cpp
struct A {
        int a;
        short b;
};
struct B:public A {
        short c;
};
// sizeof(B)结果为12 即A先padding后在加上
// 如果为8 那么当类型B对象强转为类型A时会有问题
```

通过指针类型转换获得虚表
```cpp
A a;
auto t1 = (int64_t *)(&a);
auto t2 = (int64_t *)*t1;
auto t3 = (Fun *)*(t2);
```

# 基础5：cpp中的多态与RTTI

静态多态与运行时多态

<img src="./img/rtti1.png"/>

虚表指针不指向虚表的第一项即type_info, 而是指向第二项

会生成两个析构函数

多态只能运用于指针和引用：主要是虚表变不变的问题：
- 当对象转换时调用的是static_cast 虚表变了
- 而指针转换时调用的是reinterpret_cast 虚表没变
```cpp
Test test;
Test1 testa = test; // 非多态
Test1* testb = &test; // 多态
```



获取type_info的两个方法
```cpp
std::type_info *base_type = (std::type_info *)*((int64_t *)*(int64_t *)(&a) - 1);
base_type->name();

typeid(a).name();
```

typeid返回的std::type_info删除了拷贝构造，所以只能用引用/指针接收 返回值会忽略cv限定符

dynamic_cast更安全 向上转换可以 向下转换返回nullptr

# 基础6：类型转换

static_cast<>()
- 继承但没有多态的指针以及引用可以下行转换 但是会有风险
- 多态类型转换 上行可以但没多态 下行不行
- 指针与引用都是随便转 但就是会有风险

dynamic_cast<>() 动态类型转换 更安全

const_cast<>() 去掉底层const 但是依然不能改 改就段错误 用途是函数重载避免代码重复

reinterpret_cast<type>(expression) 为运算对象的位模式提供低层次上的重新解释
- type与expression至少有一个是指针/引用
- 指针类型之间的转换可以直接来 而不需像static_cast<>()一样要用void *来当媒介

# 基础7：lambda表达式初探

类内可以声明一个引用却不给他赋值，但一定得在构造函数的初始化列表里初始化

lambda表达式的底层实现原理
```cpp
size_t sz = 10;
auto sizeComp = [sz](const string& a){ return a.size() > sz;};
// 其实是帮你创建了一个可调用的类并为你实例化了一个对象
class SizeComp {
public:
        SizeComp(size_t n):sz(n) {}
        bool operator()(const string &a) const {...} // 注意这个const
private:
        size_t sz;
};
```


lambda表达式的基础语法
```cpp
[captures](params) specifiers exception -> ret {body}
// specifiers 可选 默认是const 可以使用mutable
// exception 可选 可使用noexcept
// params 可选 c++14后可以放auto
// ret 可选 一般可以自动推导 但初始化列表推导不出来
/* captures 必须
        - 只能捕获非静态局部变量 不然也没捕获的意义 可按值,按引用或组合 按值[param] 按引用[&param] c++14后可以捕获右值 所以可以捕获地址 像这样写[param = &param]
        - 捕获发生在lambda表达式定义的时候 而不是使用的时候
        - 广义的捕获（since c++14）至此捕获列表可传右值，以及解决了一些类无法拷贝的情况如unique_ptr [r = std::move(x)]
   [this]
   [=] 捕获所有局部变量的值 包括this 但编译器会优化
   [&]
   [*this] since c++17 解决多线程问题
*/
```

# 基础8：可调用对象类型

lambda表达式存在的意义：闭包
闭包：带有上下文状态的函数
```cpp
int sz = 4;
// 主要就是这个sz
auto wc = find_if(a.begin(), a.end(), [sz](const string &s) {
        return s.size() == sz;});
```
闭包实现的3种方式：重载operator(), lambda, std::bind(绑定参数)

C++中的可调用对象：
- 函数
- 函数指针
- lambda表达式
- std::bind
- 重载函数调用运算符的类

std::function<int(int, int)>可以容纳所有


[]无任何捕获时，返回值就是函数指针


# 条款5：优先考虑auto而非显式类型声明


原因：
- 类型名太长了
- c++14之后lambda表达式中的形参也可以使用auto
- lambda表达式的返回值一定要用auto 因为底层是生成匿名类 也可以用std::function 会有性能损耗
- 与类型快捷方式有关的问题
```cpp
std::vector<int> v;
unsigned sz = v.size(); // v.size() 返回的类型在一些系统上与unsigned还是不同的 所以最好还是用auto 增加可移植性
```
- 避免因为类型写错而导致的无用的拷贝
```cpp
int a = 10;
float& b = a; // 不行
const float& c = a; // 行 创建了一个临时变量

for (const std::pair<std::string, int> &p : m) {

}
for (const auto &p : m) { // 不一样

}
```

# 基础9：CRTP与Expression Template
CRTP: 奇异递归模板模式 相对于运行时虚函数的动态多态提升了运行时性能
```cpp
template<typename Derived>
struct Base {
        void name() {
                (static_cast<Derived *>(this))->impl();
        }
};
struct D1 : public Base<D1> {
        void impl() {
        }
};
struct D2 : public Base<D2> {
        void impl() {
        }
};
template <typename Derived>
void func(Base<Derived> derived) {
        derived.name();
}
int main() {
        // 静态多态
        Base<D1> d1;
        Base<D2> d2;
        D1 d3;
        D2 d4;
}
```

表达式模板 Expression Templates：
作用：
- 延迟计算表达式，从而可以将表达式传递给函数参数，而不是只能传表达式的计算结果
- 节省表达式中间结果的临时存储空间，减少向量等线性代数计算的循环次数，从而减少整体计算的空间和时间成本

https://blog.csdn.net/HaoBBNuanMM/article/details/109740504

# 条款6：auto推导若非己愿，使用显式类型初始化

代理类就是以模仿和增强一些类型的行为为目的而存在的类。

cpp不支持两次以上的隐式类型转换

不要用引用去接临时对象的引用
```cpp
std::vector<A> func() {
        return {A(1), A(2), A(3)};
}
A &a = func()[0]; // bad 因为vector<A>的[]返回的是引用 而这资源早被释放掉了
```

vector<bool>的返回值类型是引用的代理类，所以下述操作不安全
```cpp
std::vector<bool> func() {
        return {true, true, false};
}
auto b = func()[0]; // 不安全
auto b = static_cast<bool>(func()[0]); // OK
```

# 条款8：优先考虑nullptr而非0和NULL

```cpp
// 使用auto验证一下nullptr，0，NULL分别是什么类型
auto a = 0; // int
auto b = NULL; // long 
auto c = nullptr; // std::nullptr_t

```
所以要用nullptr才能正确的调用指针版本的重载

模板推导时使用不能混用
```cpp
template<typename FuncType, typename PtrType>
decltype(auto) func(FuncType f, PtrType ptr) {
        return f(ptr);
}
bool f3(int* pi);
auto result = func(f3, nullptr); // 只能用nullptr
int a = 0;
int *p = a; // 错误
std::shared_ptr<int> sp = a; // 错误
```

# 基础10：构造函数语义学

编译器是如何完善构造函数的
- 声明时初始化的非静态成员，编译器负责将其安插进构造函数（插在前头，如果有用初始化列表初始化的会替换掉）
- 类中非静态成员未初始化且存在默认构造函数的，编译器也会安排进构造函数
- 基类的默认构造函数
- 虚表指针设置


如果编译器不需要做上述的一些事它就不会合成默认构造函数，或称其默认构造函数为trivial

class只要定义了任何一个构造函数，就都不会默认生成

基类如果不存在默认构造，子类中要手动初始化

使用using去掉子类中冗余的构造函数
```cpp
struct G : public F {
        using F::F;
        void func() {...}
}
```

# 条款7：理解特殊成员函数的生成

1. 析构
2. 拷贝构造
3. 拷贝赋值
4. 移动构造
5. 移动赋值

只有需要手动资源管理的类才会声明这些函数

只要声明了一个，其他的都应该声明出来

为了兼容c++98:1/2/3声明其中某几个不会阻止另外几个的默认生成
C++的及时止损：
- 声明1/2/3就不会生成4/5 就是 = delete掉了
- 声明4/5就不会生成2/3/4/5

移动只是建议

硬是要默认版本 可以 = default


# 条款15：尽可能的使用constexpr

将运行期的工作放到编译期来做来优化

运行期的话你要取值 计算 然后放回
而编译期优化就是直接帮你把值放那了

编译期常量
```cpp
int e = 10;
const int a = 10; // 编译期
const int b = 1 + 2; // 编译期
const int c = e; // 运行时
const int d = get_value();
```
有一些模板的声明以及数组的声明是需要使用编译期常量的
```cpp
std::array<int, 10> arr;
char carr[10];
```

constexpr: 可以保证声明出的变量是编译期常量 constexpr > const

constexpr 函数(虽然看起来constexpr很像修饰返回值): c++11
- 普通函数
        1. 函数的返回值类型不能是void
        2. 函数体只能是return expr 其中expr必须是一个常量表达式
        ```cpp
        // 这个不行
        constexpr int next(int x) {
                return ++x;
        }
        ```
        如果传给constexpr函数运行时的值，那constexpr函数会退化成普通函数
- 构造函数
        3. 构造函数初始化列表中必须是常量表达式
        4. 构造函数的函数体必须为空
        5. 所有和这个类相关的成员的析构函数必须是默认的
- 方法
        6. constexpr声明的成员函数具有const属性

c++14对constexpr函数的增强(其实就是使得不仅仅是编译期变量可以调用，运行期变量也可以调用，会帮你退化):
- 打破1，2，4，6 并且添加：函数可以修改生命周期和常量表达式相同的对象

c++17：
- if constexpr
```cpp
// 判断条件必须是编译期就能计算的
void check() {
        // 如果条件成立，else都不会编译
        if constexpr (sizeof(int) > sizeof(double)) {
                ...
        } else {
                ...
        }
}
```

# 基础11：Cpp中的异常处理以及swap & copy

stack unwinding

就是销毁try内被捕捉到的异常函数的局部变量

构造函数的try-catch
```cpp
Array(std::size_t n)
try: size_(n) {

} catch (...) {
        
}
```

不抛出保证：noexcept 关键字可以让编译器进行更好的优化
如果用了noexcept但是还是抛异常了，调用栈可能开解（直接奔溃）

移动构造 移动赋值如果不声明为noexcept，编译器是不敢用的

用法
```cpp
// 括号内的必须是编译期可计算的
void func() noexcept(true) {
}
// 还可以 noexcept(noexcept(std::swap(thing, other.thing)))
```

测试异常级别可以自己抛异常试试看


```cpp
Buffer(Buffer &&buffer) noexcept : Buffer(0) {
        swap(buffer, *this);
}

// 传右值调移动构造 传左值调拷贝构造
Buffer &operator=(Buffer buffer) { 
        swap(buffer, *this);
        return *this;
}
```

swap具体看efcpp版本

# 条款14：如果函数不抛出异常请使用noexcept

c++11中内存释放函数和析构函数都是隐式noexcept，除非手动声明为noexcept(false), 总之就是析构函数别抛异常

如果一些特殊函数没声明为noexcept的话，一些标准库里的函数是不敢用的，所以会降低效率

# 条款10：优先考虑限域enum而非未限域enum

enum中的每个元素的底层类型都是一个由编译器决定的整型（元素的值可以手动设置，且可以相同）

未限域enum的问题
- 作用域是全局的
- 可隐式转换为整型
```cpp
enum School {
        teacher,
        student
};
int main() {
        School stu = student;
        int i = student;
        School stu1 = 10; // 不行
        School stu2 = static_cast<School>(10); // 可以 但未定义行为
}
```
- 通常情况下无法前置声明，因为不知道要分配多大的空间，除非在声明与定义处都指定其整形的类型

限域enum
```cpp
enum class EnumInt {
        e1 = 1
};
int main(void) {
        EnumInt ei = EnumInt::e1;
}
```
解决了上述未限域的几个问题，默认整形是int

未限域的enum的一个用处
```cpp
UserInfo uInfo{"hkl", "7xxx@email.com", 10};
auto val = std::get<1>(uInfo); // 比较含糊这个1
auto val_ = std::get<uiEmail>(uInfo); // nice 而用限域的要强转 写一大串
```

当然也可以把类型转换放到一个函数里
```cpp
template <typename E>
constexpr typename std::underlying_type<E>::type
toUType(E enumerator) noexcept {
        return static_cast<typename std::underlying_type<E>::type>(enumerator);
}
// c++14
template <typename E>
constexpr auto
toUType_(E enumerator) noexcept {
        return static_cast<std::underlying_type_t<E>>(enumerator);
}
```

c++17 可以用列表初始化有底层类型的枚举对象（限域，不限域都可以）
```cpp
enum class Index : int {}; // 应用场景是需要一个新整数类型时
```

c++20 可以使用using打开限域enum
```cpp
switch (c) {
        using enum Color;
        case Red : return "Red"; // 这边就可以不用限域运算符了
        ...
}
```

# 基础12：友元相关

可以做友元的3个东西
- 全局函数
- 类方法
```cpp
class Building; // 这边先声明一下
class LaoWang {
public:
        void visit(); // 会访问Building的私有成员 故需要设置为友元
private:
        Building *building_; // 这边不能是值成员，因为目前编译器还无法确定Building类的大小，故无法分配空间

        // 甚至涉及到Building函数的定义也不能在这边写，原因如上
};
class Building {
        friend void LaoWang::visit();
        ...
};
```
- 类


# 条款11：优先考虑使用delete函数而非使用未定义的私有声明

c++98中如果你想禁用拷贝构造与拷贝赋值，你得把其声明写在private中，然后不要定义
但这是不安全的，因为友元是可以访问的到你的，这时候访问到你，但你却没有实现会发生link error

用=delete，并写在public中(否则报错信息不明确）是最好的实践

delete可以用来删除任何函数
```cpp
bool isLucky(int n);
bool isLucky(char) = delete;
isLucky('a); // 报错
```

delete可以禁止一些模板的特化
```cpp
template <typename T>
void processPointer(T* ptr);

template <>
void processPointer<void>(void*) = delete;
// 在类内的模板特化禁止不能写在private中 只能用delete
```


# 条款12：使用override声明重写函数

重写函数要满足的充分条件
- 基类函数必须是virtual
- 函数名（除了析构函数），形参类型，常量性必须完全一致
- 返回值和异常类型与派生类型必须兼容（注意这个兼容）
- 引用限定符必须一致

如果你写override的话，若上述条件没全部满足时是会给你报错的

不写override的时，显的很含糊，编译器可能不理解你的意图

final 标识符：可以防止虚函数被重写，类被继承

引用限定符（c++11）：
```cpp
class Widget {
public:
        // 左值Widgets会调用
        void func() & {
                cout << "Left value" << endl;
        }
        // 右值Widgets会调用
        void func() && {
                cout << "Right value" << endl;
        }
}
```

# 条款13：优先考虑const_iterator而非iterator

zeal这个软件可以下载一些文档

c++11之前，标准库不提供vector的cbegin, cend的成员函数
虽然iterator可以用static_cast强转
而c++11之前vector的insert函数只支持iterator而不支持const_iterator
const_iterator又无法转成iterator

c++11之后，c++14之前，没有对于非成员函数的cbegin，cend的实现，但可以通过向begin或end函数中传入const容器获得

