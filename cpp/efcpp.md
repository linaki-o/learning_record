构造函数声明为explicit更好

当有新对象被创建时=符号调用的是copy构造函数，否则调用的是copy赋值函数

函数调用时 如果以by value的方式传递对象则会调用拷贝构造函数，开销较大，pass-by-reference-to-const是更好的选择
> 视C++为一个语言联邦
> 尽量以const, enum, inline替换#define
好处：
- 容易debug
- 使用常量只有一份而define可能有好多份


新式编译器：static const 类内成员需要在类内声明一次值，在类外即实现文件还得再定义一次，不过不用赋值
define 没有任何封装性

旧式编译器：头文件声明，实现文件定义且赋值

> 尽可能使用const

const在*外指指向的值为常量

const iterator是T *const
stl中的const_iterator指的是const T*


把返回值设置为const也是挺好的 特别是operator*()

const成员函数 常量性不同的成员函数是可以被重载的
- bitwise const
- logical const mutable

const对象只能调用const成员函数 const对象大多用于passed by pointer(reference)-to-const

cpp 是by value 返回对象的


当代码极大重复时，可以让non-const函数调用const函数，只不过要做几次转型

> 确定对象使用前已先被初始化
cpp规定对象的成员变量初始化动作发生在进入构造函数本体之前, 这对内置类型是不确定的 最好用member initialization list来构造 const 或者references 一定要以这种方式来赋值


class的成员变量总是以声明次序被初始化
 
non-local static对象的初始化次序无明确定义 为了解决这个问题可以给它们分别分配一个专属函数，然后函数返回一个reference 这就是singleton（非常好的pratices) 即用local static替换掉了non-local static 当在多线程情况下还是会有race

> cpp默默编写并调用了的函数

有哪些（都是public且inline）：
- copy构造函数
- copy赋值函数
- 析构(非virtual)
- 构造(用于调用base classes和non-static成员变量的ctor与dtor)

在合法的情况下，default生成的copy构造函数与copy赋值函数是会 内置类型直接复制 有copy函数的调copy函数

> 若不想自动生成应该明确拒绝

用private且不实现：
- 当member函数或friend函数调用时会undefined reference错误（没错可以不实现，有空可以看看底层是怎么实现的）
- 这时候是link错误

创建一个uncopyable的基类，然后private继承
- compile错误 better

> 为多态基类声明virtual析构函数

当derived class对象经由一个base class指针被删除，而base class的dtor是个non-virtual，则derived class的析构不会调用

否则不用加 否则vptr也是要占一定空间的

析构的顺序是由子向基


> 别让异常逃离析构函数
> 绝不在构造和析构过程调用virtual函数

base class构造期间virtual函数绝不会下降到derived classes阶层

因为这时derived class的成员变量尚未初始化 不只virtual函数会被这样解释，若使用runtime type information如dynamic_cast和typeid也会把对象视为base class

对象在derived class构造函数开始执行前不会成为一个derived class对象

析构函数也是一样的道理

> 令operator=返回一个reference to *this

> 在operator=中处理 自我赋值

方法：
- 多加一层验证
- 没完全结束之前不删原来的资源(异常安全)
- copy and swap(注意把参数类型给改成非引用，这样就是把copying动作从函数本体内移至函数参数构造阶段)

> 复制对象时勿忘其每一个成分

每一个都要拷贝

当为derived class编写copying函数的时候也要小心的复制其base class成分

```cpp
Derived_Ctor():Base_Ctor(rhs); // 拷贝构造函数
Ctor::operator=(rhs); // 拷贝赋值函数
```

有重复代码就创建一个init函数

> 以对象来管理资源

Resource Acquisition Is Initialization;RAII

c++11 以后智能指针好像就可以处理动态分配数组了

> 在资源管理类中小心copying行为

资源copying的行为决定了RAII对象的copying行为

解决方法：
- 禁止复制
- 引用计数法
    - 可以通过指定删除器来特定化行为（例如遇到锁的情况）
- 复制底部资源 深度拷贝
- 转移底部资源的所有权

> 在资源管理类中提供对原始资源的访问

显式转化
    - get() 返回原始指针
隐式转化
    - 重载操作符operator->和operator*
    - 隐式转换函数
    ```cpp
    也会出现错误 dangle错误
    Font f1(getFont());
    ...
    FontHandle f2 = f1;
    ```
> 成对使用new和delete时要采取相同形式

数组所用的内存通常还包括 数组大小 的记录

小心不要对数组形式做typedefs动作 否则在new delete的时很危险

> 以独立语句将newed对象置入

```cpp
void processWidget(std::shared_Ptr<Widget>(new Widget), priority())
// 我实验过代码都是priority先执行
```

> 让接口容易被正确使用 不易被误用

```cpp
Date(int month, int day, int year); // 用wrapper types会更好些
```

应该尽量令你的types的行为与内置types一致

任何接口如果要求用户必须记得做某些事情，就是有着不正确使用的倾向

当一个接口返回指针时 很有可能这个指针在后期用户没有合理的释放

"cross-DLL problem"

> 设计class犹如设计type

考虑条款19的主题

> 宁以pass-by-reference-to-const替换pass-by-value

以by reference方式传递参数也可以避免slicing问题：当一个derived class对象以by value方式传递并视为一个base class对象，base class的copy构造函数会被调用，而切割调一切derived class对象的特征，只留下一个base class对象(实验起来有点奇怪)

> 必须返回对象对象时，别妄想返回其reference

任何时候看到一个reference声明式，都应该问自己它的另一个名称是什么

good practice
```cpp
inline const Rational operator*(const Rational& lhs, const Rational& rhs) {
    return Rational(lhs.n * rhs.n, lhs.d * rhs.d);
}
```

> 将成员变量声明为private

好处
- 语法一致性 每一个public接口都是函数会方便用户
- 封装 方便开发者以后扩展 而不需要改变用户的程序
- 当一个protected成员变量被取消，所有使用它的derived classes都会被破坏

> 宁可以non-member, non-friend替换member函数

越多东西被封装 我们就有越大的弹性去变化它

封装性可以根据可以访问这块数据的函数数量来粗略计算

将non-member,non-friend函数放在相同的namespace中

cpp标准程序库有数十个头文件，每一个头文件声明std的某些机能

> 若所有参数皆需类型转换，请为此采用non-member函数
```cpp
result = 2 * oneHalf // 不太得行
```

只有当参数被列于参数列内，这个参数才是隐式类型转换的合格参与者
```cpp
const Rational operator*(const Rational& lhs,
                         const Rational& rhs) {
    return Rational(...);
}
```

无论何时如果你可以避免friend函数就该避免，因为就像真实世界一样，朋友带来的麻烦往往多过其价值

> 考虑写出一个不抛异常的swap函数

引用初始化后就不可改变

pimpl手法 pointer to implementation

模板特化
```cpp
// 目前还不能通过编译 因为pimpl是private的
namespace std {
    template<>
    void swap<Widget>(Widget& a, Widget& b) {
        swap(a.pimpl, b.pimpl);
    }
}
```
面对class时的good practice
```cpp
// 与STL容器有一致性
class Widget {
public:
    void swap(Widget& other) {
        using std::swap;
        swap(pimpl, other.pimpl);
    }
};
namespace std {
    template<>
    void swap<Widget>(Widget& a, Widget& b) {
        a.swap(b);
    }
}
```

linux下头文件放在usr/include中

```cpp
// 企图偏特化一个function template是行不通的 只能是class templates
namespace std {
    template<typename T>
    void swap<Widget<T>>(Widget<T>& a, Widget<T>& b) {
        a.swap(b);
    }
}

// 重载
namespace std {
    template<typename T>
    void swap(Widget<T>& a, Widget<T>& b) {
        a.swap(b);
    }
}

```

上面的重载行不通，因为可以全特化std内的template，但是不可以添加新的templates到std里头

class与class template的good practice
```cpp
namespace WidgetStuff {
    template<typename T>
    class Widget{...};

    template<typename T>
    void swap(Widget<T>& a, Widget<T>& b) {
        a.swap(b);
    }
}
// C++的name lookup rules会找到WidgetStuff的专属版本
```

特化：如果你能对某一个功能有更好的实现，那么就该听你的
全特化：限定死模板实现的具体类型
偏特化：如果这个模板有多个类型，限定其中一部分
```cpp
template <class T1, class T2>
class Test {};
template <>
class Test<int, char> {};
template <class T2>
class Test<int, T2> {};
```

函数不能偏特化 因为偏特化的功能可以通过函数的重载完成

C++ name lookup rules:
- global作用域或T所在之命名空间内任何T专属的swap
- std内的swap (要包含一个using声明式，以便让std::swap在你的函数内曝光可见，然后不加任何namespace修饰符，赤裸裸的调用swap)
    - 特化的
    - 一般的

对于classes（而非template）也请特化std::swap
"swap的成员版绝不可抛出异常" 因为swap的一个最好的应用是帮助classes和template提供强烈达到异常安全性保障

高效率的swaps几乎总是基于对内置类型的操作，而内置类型上的操作绝不会抛出异常

# 实现
问题：
1. 太快定义变量可能造成效率上的拖延
2. 过度使用转型可能导致代码变慢又难维护又容易错
3. 返回对象内部数据的handles可能会破坏封装且dangling handles
4. 未考虑异常会导致资源泄漏和数据败坏
5. 过度inlining代码膨胀
6. 过度耦合导致build times冗长

> 尽可能延后变量定义式出现时间

尝试延后定义直到能给它初值实参为止 可以避免不必要的构造析构复制

循环的话要分析一下
- 效率
- 程序可理解性和易维护性

> 尽量少做转型动作

```cpp
// 旧式
(T)expression
T(expression)
// 新式
const_cast<T>(expression) // 移除常量性
dynamic_cast<T>(expression) // 安全向下转型 "可能耗费重大运行成本"
reinterpret_cast<T>(expression) // 低级转型 不可移植
static_cast<T>(expression) // 强迫隐式转换 int转为double
```

新式的好处
- 很容易辨别
- 编译器好排错

任何一个类型转换（不论是通过转型操作而进行的显式转换或通过编译器完成的隐式转换）往往真的令编译器编译出运行期间执行的码

"C++中单一对象可能拥有一个以上的地址" 关键看对象在Cpp中如何布局的

```cpp
class Widget {
public:
        int a;
        Widget(int a):a(a) {
        }
        virtual void test(int b) {
                cout << "Widget" << " " << this << " " << a << endl;
                a = b;
        }

};
class DWidget : public Widget {
public:
        DWidget(int a):Widget(a) {}
        virtual void test(int b) {
                // static_cast<Widget>(*this).test(b); 在副本上操作了 不符合我们的意图
                Widget::test(b);
                cout << "DWidget" << " " << this << " " << a << endl;
        }

};
```

dynamic_cast<>()的替代法：
1. 用子类容器如vector
2. 提供virtual函数（即使在基类中是空实现）

尽可能隔离转型动作，封装起来


> 避免返回handles指向对象内部成分

坏处：
- 破坏封装
- 可能导致对象dangling 特别是因为函数返回临时对象时

> 为异常安全而努力是值得的

带有异常安全性的函数会:
- 不泄露任何资源（以对象管理资源）
- 不允许数据败坏
    - 异常安全函数提供三个保证之一
        - 基本承诺：任何事物仍然保持在有效状态下。没有对象或数据结构会因此而败坏
        - 强烈保证: 要么成功，要么保持原样 智能指针 CopyAndSwap
        - 不抛掷保证: 承诺绝不抛出异常，作用于内置类型身上的操作都提供nothrow保证

"意想不到的函数：set_unexpected"

任何使用动态内存的东西如果无法找到足够内存以满足需求，通常便会抛出一个bad_alloc异常，所有nothrow保证很难实现

强烈保证也不好实现 特别是需要所有函数都合规

异常安全只有“有和没有”

> "透彻了解inlining的里里外外"

inline造成的代码膨胀亦会导致额外的换页行为，降低TLB的击中率

inline只是对编译器的一个申请，不是强制命令

可以隐喻提出（将函数定义于class定义式内），也可以明确提出

> 将文件间的编译依存关系降至最低

```cpp
// person.h
#include "date.h"
class Person {
public:
private:
    Date theBirthDate;
};
// 不好 当date.h改变时所有包含Person class的文件都得重新编译
```

而 main.cpp 依赖于 person.h。 那么基本上，不管你是怎么改 person.h， 甚至就算是 touch 了一下， 大部分依赖管理器也会让编译器重新编译一次，区别可能仅仅是好一点的编译器很快发现其实 person.h 根本没有变化， 编译时间稍微短一些而已。

为什么不将定义与声明分开来呢？ 不行, 编译器必须在编译期知道对象的大小
```cpp
// 不得行 编译器得询问class定义式，即class定义式需要列出实现细目
int main() {
    int x;
    Person p;
}
```

可以用pimpl idiom来实现

"以声明的依存性替换定义的依存性" 现实中让头文件尽可能自我满足，万一做不到则让它与其他文件内的声明式相依(可以实验一下)

person.h 里面只是包含了接口信息，具体的实现挪到了PersonImpl 这个类。 由于 Person 对 PersonImpl 的引用是一个指针， 而指针大小在同一平台是固定的。 所以 person.h 根本就不需要包含  ( 注意  person.cpp 需要包含 personimpl.h， 因为需要具体使用到 PersonImpl 的函数）。

这时依赖关系变为：
```
main.cpp -> person.h
person.cpp -> person.h, personimpl.h
personimpl.cpp -> personimpl.h
```


不要Premature optimization

实际上一个person.h 一个person.cpp就好了

> 确定你的public继承塑模出is-a关系

在编译期排错比在运行期排错好

适用于base classes上的也一定适用于derived classes上

> 避免遮掩继承而来的名称

编译器看到名称会从小到大的查找各个作用域

```cpp
class Base {
public:
        virtual void mf1() = 0;
        virtual void mf1(int a) {
                cout << "B mf1" << a << endl;
        }
        virtual void mf2() {
                cout << "B mf2"<< endl;
        }
        void mf3() {
                cout << "B mf3"<< endl;
        }
        void mf3(double a) {
                cout << "B mf3" << a << endl;
        }
};
class Derived: public Base {
public:

        // using Base::mf1; 加上这两句就可见了 也可以通过forwarding functions
        // using Base::mf3; 
        virtual void mf1() {
                cout << "D mf1" << endl;
        }
        void mf3() {
                cout << "D mf3" << endl;
        }
        void mf4() {
                cout << "D mf4" << endl;
        }
};

int main() {
        Derived d;
        int x = 9;
        d.mf1();
        d.mf1(x); // 错误 因为函数遮掩
        d.mf2();
        d.mf3();
        d.mf3(x); // 错误 因为函数遮掩
        return 0;
}
```

> 区分接口继承和实现继承

```cpp
class Shape {
public:
    // pure virtual函数的目的是为了让derived classes只继承函数接口
    // 但是可以提供定义 但要通过ps->Shape::draw(); 来使用
    virtual void draw() const = 0; 
    // impure virtual函数的目的是继承接口以及缺省实现 可以进一步实现接口与缺省实现分离
    virtual void error();
    // non-virtual函数的目的是为了令derived classes继承函数的接口以及强制性实现
    void error();
};
```

> "考虑virtual函数以外的其他选择"

**Non-Virtaul Interface实现Template Method模式 其实就是wrapper**

```cpp
#include <iostream>
using namespace std;
class Base {
public:
        void func() const {
                pvfunc();
        }
private:
        // const函数只能调用const函数
        virtual void pvfunc() const {
                cout << "Base" << endl;
        }
};
class Derived:public Base {
};
int main() {
        Derived d;
        // 居然可以继承private virtual函数，也可以在基类重写
        d.func();
        return 0;
}
```

**通过Function Pointers实现Strategy模式**

就是NVI方式中的对象的virtual函数换成函数指针，然后构造函数赋值，这样同一个类的对象可以有不同的行为

主要问题就是函数指针的外部函数的问题，如何定义，如何访问对象的private属性

**通过std::function完成Strategy模式**

> 绝不重新定义继承而来的non-virtual函数

```cpp
#include <iostream>
using namespace std;
class Base {
public:
        void mf() {
                cout << "Base" << endl;
        }
};
class Derived: public Base {
public:
        void mf() {
                cout << "Derived" << endl;
        }
};

int main() {
        Derived x;
        Base* pB = &x;
        pB->mf(); // 会有不同的行为
        Derived* pD = &x;
        pD->mf();

        return 0;
}
```

因为non-virtual函数B::mf和D::mf都是静态绑定的

virtual函数是动态绑定

> 绝不重新定义继承而来的缺省参数值(即virtual函数的)

virtual函数是动态绑定，而其缺省参数是静态绑定的

```cpp
class Circle: public Shape {
public:
    virtual void draw(ShapeColor color) const;
    /*
        当以对象调用此函数时，一定要指定参数  
        当以指针调用此对象时，可以不指定参数
        这就是动态绑定和静态绑定的原因
    */
}
```

动态类型: 指针所指向的对象
静态类型: 指针的类型
```cpp
#include <iostream>
using namespace std;
class Base {
public:
        enum ShapeColor {Red, Green, Blue};
        virtual void mf(ShapeColor color = Red) {
                cout << "Base " << color << endl;
        }
};
class Derived: public Base {
public:
        virtual void mf(ShapeColor color = Green) {
                cout << "Derived " << color << endl;
        }
};

int main() {
        Base *pd = new Derived;
        pd->mf(); // 返回的是Derived 0！
        return 0;
}
```

```cpp
#include <iostream>
using namespace std;
class Base {
public:
        enum ShapeColor {Red, Green, Blue};
        virtual void mf(ShapeColor color = Red) {
                cout << "Base " << color << endl;
        }
};
class Derived: public Base {
public:
        virtual void mf(ShapeColor color) {
                cout << "Derived " << color << endl;
        }
};

int main() {
        Base *pd = new Derived;
        pd->mf();
        Derived *pd1 = new Derived;
        pd1->mf(); // 编译期直接报错
        return 0;
}
```

引用也存在上述问题

C++坚持上述方式是因为运行期效率

我们可能会考虑让所有的derived类都定义相同的缺省参数
但当base的缺省参数要改变时就要更改很多 所以用到上一条款的方法