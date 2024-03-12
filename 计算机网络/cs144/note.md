# Intro
 
target:
1. To learn how the Internet works.
2. To learn why the Internet was designed this way.
3. To learn how to use the Internet.
4. To build some significant pieces of the Internet.
5. To learn the underlying principles and technologies of networking.

the 1st week is about Basic Principles, then (2) Transport, (3) Packet Switching, (4) Congestion Control, (5) Routing, (6) Lower Layers, (7) Applications

# Lab Checkpoint 0: networking warmup [2 and 6 hours to complete already 2h]

The “smtp” service refers to the Simple Mail Transfer Protocol

netcat (server) or the telnet (client)

the only thing the Internet really does is to give its “best effort” to deliver short pieces of data, called Internet datagrams, to their destination. 

docker容器启动不了的问题
```
感觉就是因为我少加了-it浪费了一下午的时间
https://cloud.tencent.com/developer/article/1341775
https://www.zhihu.com/question/442341345
```

docker挂载目录
```bash
docker run --name test -it -v /home/xqh/myimage:/data ubuntu /bin/bash
```


In practice datagrams can be (1) lost, (2) delivered out of order, (3) delivered with the contents altered, or even (4) duplicated and delivered more than once. It’s normally the job of the operating systems on either end of the connection to turn “best-effort datagrams” (the abstraction the Internet provides) into “reliable byte streams” (the abstraction that applications usually want).

The basic idea is to make sure that every object is designed to have the smallest possible public interface, has a lot of internal safety checks and is hard to use improperly, and knows how to clean up after itself.

We want to avoid “paired” operations (e.g. malloc/free, or new/delete)

std::string_view
```

生命周期： std::string_view 仅仅是对已存在的字符串数据的一个非拥有引用，因此需要确保原始字符串的生命周期长于 std::string_view 的使用。如果原始字符串被销毁，那么 std::string_view 就会变成悬空引用，导致未定义的行为。

不拥有内存： std::string_view 不负责管理字符串的内存，因此不能对其进行修改。它只是提供了一个不可变的视图。

不添加空字符： std::string_view 不会在字符串的末尾添加空字符 '\0'，因此可以处理包含空字符的字符串。

性能： std::string_view 是一个轻量级的对象，因为它只是一个指向原始数据的指针和长度的组合，所以传递和复制 std::string_view 对象的成本通常很低。

不检查边界： std::string_view 不会检查索引是否越界，因此使用它时要确保不会访问超出字符串范围的位置，否则会导致未定义的行为。
```
std::shared_ptr
```
当你使用 std::shared_ptr 的移动构造函数时，只需要传递一个同类型的 shared_ptr 对象，该对象的引用计数会增加，而原来的 shared_ptr 对象的引用计数会减少。
```


std::span
```
C++20引入了std::span作为一种语法糖，用于表示连续内存范围。它提供了一种轻量级的、非拥有式的、零开销的方式来引用数组或其他连续内存块。std::span可以用于传递数组片段给函数，或者在函数内部对连续内存进行操作，而无需进行内存拷贝。
```

concept
```cpp
模板一般会有一些限制，为了选中最佳的函数，会对模板参数有些需要

这些需要就被称为concepts 是在编译期的一种预测 作为模板接口的一种限制

// Declaration of the concept "Hashable", which is satisfied by any type 'T'
// such that for values 'a' of type 'T', the expression std::hash<T>{}(a)
// compiles and its result is convertible to std::size_t
template<typename T>
concept Hashable = requires(T a)
{
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};
struct meow {};
// Constrained C++20 function template:
template<Hashable T>
void f(T) {}

int main()
{
    using std::operator""s;
 
    f("abc"s);    // OK, std::string satisfies Hashable
    // f(meow{}); // Error: meow does not satisfy Hashable
}
```

std::optional
```
since C++17(当clangd识别不了时用CMAKE把c++版本改成17)
std::optional 是 C++ 标准库中的一个类模板，它表示一个可能存在或不存在的值。它类似于指针，但它可以安全地表示空值，而不会导致空指针异常。

has_value()

The size of std::optional in C++ can vary depending on the implementation and the type it holds. Typically, a std::optional may have a size larger than the size of the contained type due to internal bookkeeping

内部有一个bool类型数据对象故 size=sizeof(YourType) + 1 但还需考虑对齐的情况

有加强版：sizeof(tiny::optional<double>) == sizeof(double)
```

exception.hh demangle
```cpp
ABI（应用程序二进制接口）是一组约定，它定义了不同组件（例如编译器、操作系统和库）如何交互。它指定了以下内容：

数据类型的表示形式（例如大小、对齐方式和字节序）。
函数调用约定（例如参数传递和返回值）。
异常处理机制。
内存布局约定（例如堆栈和堆的组织方式）。
ABI 对于确保不同组件能够协同工作至关重要。它允许在不同的编译器、操作系统和库之间链接和调用代码。

作用：获取取消修饰的名称。

```

abi::__cxa_demangle 函数的作用：
```
作用：取消 C++ 名称的修饰。
C++ 编译器会在编译过程中对函数和变量名称进行修饰，以便在程序中唯一标识它们。修饰后的名称称为 修饰名称，它包含有关函数或变量类型和参数的信息。
```

from `/etc/services`, e.g., "http" is port 80

**An in-memory reliable byte stream**
```
还好可以通过测试用例分析一下

其实就是队列

peek: 先进先出 把队头的拿出去
pop: 小于截断 大于我也是只把一个给扔出去
push当超过容量后是怎么操作 是阻塞 还是截断 然后直接退出: 截断然后直接退出

数据结构要求：
- 数组不行，因为要是用pop的话每次都得移动元素

```

用了一些cmake的拓展确实还挺好使的, 可以借此机会再学一下cmake
> 测试框架
```
TestHarness 最大的基类 可以执行TestStep 并记录每一次执行
    ByteStreamTestHarness 非常明显是什么作用了
    ReassemblerTestHarness
        基类的obj_现在变成{ Reassembler { ByteStream { capacity } } }

TestStep 抽象类 它的模板就是execute的参数类型
    Action 依然是抽象类 把一些描述性的函数实现了
        Push主要函数 execute
        Insert Action<Reassembler>
    ReassemblerTestStep:TestStep<Reassembler>
        成员step_也是TestStep byte_stream_test_step
        execute step_.execute( r.reader() )
    Expectation 抽象类 与Action是一类的
        ExpectNumber execute 检测值是否一致
            BytesPushed 主要函数value
            BufferEmpty
        ReadAll 自己execute完了还会让BufferEmpty检测一下还有没有

```
> 收获
```
代码是通过下行转换来获取Reader和Writer的 static_cast<const Reader&>( *this );
不过它通过保证子类没有其他的元素来保证了转换的安全性，并且也没有虚函数

The first thing to say is that overloading std::to_string for a custom type is not allowed. We may only specialise template functions and classes in the std namespace for custom types, and std::to_string is not a template function. 得用查表法


string_view比string快的多 尽量用

```

> 自我总结
```
没认真看测试代码与项目源码 导致问题的解决方法想偏了

不过后面看项目源码的同时还是收获良多

得仔细看测试代码 不然实现的不清不楚

2024/02/20 
- CMake的构建框架是了解了一遍 大概知道是在干啥了 差测试模块
- 该自己写的部分看完了 做了一点优化明天做一点实验
- 明天把测试方面的代码与构建框架看一下
2024/02/21
- 得debug看看我的操作是哪里错了 原来是因为string_view的性质的问题 解决了
- 测试框架看的差不多了 明天改一下代码就可以继续了
```





# Lab Checkpoint 1: stitching substrings into a byte stream [来来回回搞了一周]

mtr - a network diagnostic tool

TCP sender: 拆分 byte stream成小段使其能放到datagram中
TCP receiver: 
- 接收datagrams然后变成可依赖的byte stream, 可以被应用通过socket来读
- datagrams可能会顺序不当 丢失 重复 得保证把这些段正确的重组


Reassembler:(本part需要完成)
- 作用重组
- (output .writer()) 顺序合适的数据通过其写到流里
- 存储早到的数据
- The Reassembler should discard any bytes that lie beyond the stream's available capacity (i.e., bytes that couldn't be written even if earlier gaps get filled in).(还不是很懂，看看测试用例)(是马上就扔)
- 数据首先存在Reassembler内部，当排序成功后输入到ByteStream
- The Reassembler should close the stream after writing the last byte.
- overlapping的情况有点棘手啊 如果调用者提供了关于同一索引的冗余知识，**重组器应该只存储该信息的一份副本。** 应该存储每一份的副本 写要及时直接写 
- 问题是如果写完后开始处理索引2了这时来了个0的说overlap该怎么处理
- 或者来一个完全不一样的overlap怎么处理  假设不存在
- 错序+超capacity不pending
- is_last()在任何情况下都会生效 并不是马上关 特别是当错序pending时
- 但是好像不一定一定要按0,1,2,3,...顺序存
- 插入时若capacity为0 直接扔 否则截断
- 情况有些多啊 得再看看材料
- 应当存储
    - 超出流可用容量的字节。 这些应该被丢弃。 重组器不会存储任何无法立即推送到字节流的字节，或者一旦知道较早的字节即可推送到字节流的字节。

测试用例:
- reassembler_single.cc
```
就是简单的插入 当is_last()时就关闭不管你插入多少，有没有插入
检查data的size
乱序不插 一定是从0开始
冗余不插
```
- reassembler_cap.cc
```
**索引0后2和4可以直接插 或者这个索引是的字符的数量 明白了!!!**


超限能pending多少pending多少

pending的数量加还没到的数量一定小于capacity

当知道索引的含义时，完全是可以预估前面有多少的

overlapping 只会变长

前面还没来的数据包的大小也得考虑 不能只考虑当下

要处理好多出来的部分

is_last 且 pending number 为0才能关闭
```
- reassembler_seq.cc
```
ovelapping的片段连接了两段的情况
```
- reassembler_dup.cc
```
一些前多余数据包的处理
```
- reassembler_hole.cc
```
判断什么时候finish的时机
一次is_last()+没有pending+明确没有后面没别的内容了

overlap 直接超过之前先到的数据包
```
- reassembler_overlapping.cc
```
Overlapping unassembled section, not resulting in assembly
现在我用的是优先队列来处理先到的情况 感觉这种数据结构处理不了

可以在pending 那边判断一下

后面优化时可以试试把hashmap换成字符串
```

Use “defensive programming”—explicitly check preconditions of functions or invariants, and throw an exception if anything is ever wrong.

modularity

typo: 打字错误

std::derived_from 
```
是 C++20 中的一个类型特征(trait)，用于检查一个类是否是另一个类的派生类。具体来说，std::derived_from<A, B> 会在编译时检查，如果 A 是 B 的公有派生类，它将评估为 true，否则为 false。 模板元编程
```

cmake是有不同的target的，执行不同的target会在不同的文件夹下调用make

> 自我总结
```
太依赖测试用例了 讲义没认真看 导致浪费了时间 写的还很乱

花的时间稍微有点多 做前得自我设定一下时间了

还有deque居然可以随机取？得了解一下底层实现了

be easy
```

deque
```
deque容器和vector容器最大的差异:
一在于deque允许使用**常数项时间**对头端进行元素的插入和删除操作

deque没有容量的概念，因为它是动态的以分段连续空间组合而成，随时可以增加一段新的空间并链接起来. 而vector就是连续的空间

也因此，deque没有必须要提供所谓的空间保留(reserve)功能. 虽然deque容器也提供了Random Access Iterator,但是它的迭代器并不是普通的指针， 其复杂度和vector不是一个量级，这当然影响各个运算的层面

除非有必要，我们应该尽可能的使用vector，而不是deque。对deque进行的排序操作，为了最高效率，可将deque先完整的复制到一个vector中，对vector容器进行排序，再复制回deque

array 无法成长，vector虽可成长(成长也是假想)

deque是由一段一段的定量的连续空间构成。一旦有必要在deque前端或者尾端增加新的空间，便配置一段连续定量的空间，串接在deque的头端或者尾端。deque最大的工作就是维护这些分段连续的内存空间的整体性的假象，并提供随机存取的接口，避开了重新配置空间，复制，释放的轮回，代价就是复杂的迭代器架构。
既然deque是分段连续内存空间，那么就必须有**中央控制**，维持整体连续的假象，数据结构的设计及迭代器的前进后退操作颇为繁琐。deque代码的实现远比vector或list都多得多。 deque采取一块所谓的map(注意，不是STL的map容器)作为主控，这里所谓的map是一小块连续的内存空间，其中每一个元素(此处成为一个结点)都是一个指针，指向另一段连续性内存空间，称作缓冲区。缓冲区才是deque的存储空间的主体。
```

合并分支
```bash
git checkout main
git merge dev
git branch -d dev
```

# Lab Checkpoint 2: the TCP receiver [S:2024/03/03 30min 25min 1h 20min 1h E:2024/03/08]

The TCPReceiver **receives messages from the peer’s sender (via the receive() method)** and **turns them into calls to a Reassembler**, which eventually **writes to the incoming ByteStream**. Applications read from this ByteStream

TCPReceiver的response:
- the index of the **first unassembled** byte, which is called the “acknowledgment number”or **ackno.** This is the first byte that the receiver needs from the sender.
- the **available capacity** in the output ByteStream. This is called the **window size**

window:
- left edge: ackno
- right edge: ackno + window size

**The hardest part** will involve thinking about how TCP will represent each byte’s place in the stream—known as a “sequence number.”

This week, you’ll implement the **receiver** part of TCP, responsible for receiving messages from the sender, reassembling the byte stream (including its ending, when that occurs), and determining that messages that should be sent back to the sender for acknowledgment and flow control

acknowledgment: sender要发或重发的next byte的索引
Flow control: sender应该发多少

Reassembler接收的参数stream index的类型是64bit的，而因为In the TCP headers, however, space is precious, and each byte’s index in the stream is represented not with a 64-bit index but with a 32-bit **sequence number**, or **seqno**.

而Streams in TCP can be arbitrarily long, 可能不仅仅是2^32bytes, 所以得处理一下, 解决办法：环形(取模)

**TCP sequence numbers start at a random value**: To improve robustness and avoid getting confused by old segments belonging to earlier connections between the same endpoints, TCP tries to make sure sequence numbers can’t be guessed and are unlikely to repeat. 

Initial Sequence Number (ISN). This is the sequence number that represents the “zero point” or the SYN (beginning of stream).


The logical beginning and ending each occupy one sequence number: SYN (beginning-ofstream) and FIN (end-of-stream) control flags are assigned sequence numbers. **Keep in mind that SYN and FIN aren’t part of the stream itself and aren’t “bytes”—they represent the beginning and ending of the byte stream itself**

seqnos在TCP segment的Header里

两个方向两个流seqnos, ISN是不同的

**Sequence Numbers, Absolute Sequence Numbers, Stream Indices是三个不同的概念，详细看图里 很重要 不然会很多bug**

将Sequence Number包装为一个类，Wrap32，类内有方法可以转换到ASN(uint64_t)

> Job one. 实现Wrap32
uint32_t类型当超出限制时自己就会取模了

uint64_t % 2^32 其实与 uint64_t强转uint32_t的作用是一样的 都是只保存最后32个bit

unwrap: Convert seqno → absolute seqno.
seqno 1 ISN 2^32 - 2 可以对应 3, 总数 + 3, ... 

The wrap/unwrap operations should preserve offsets—two seqnos that differ by 17 will correspond to two absolute seqnos that also differ by 17.

老是抓不住要点

好多数值计算的细节

uint64_t const dis = raw_value_ - ckpt32.raw_value_;
先在32bit上计算出来了再类型转换即增加位数

```c
int main() {
        uint32_t a = 4, b = 6;
        uint64_t c = b - a, d = a - b;
        printf("%ld %ld", c, d); // 设差值为x，则一个是x，一个是2^32 - x
        return 0;
}
```

利用异常机制来测试


c++里的math.h有min函数

> Job two: Implementing the TCP receiver

CMAKE的测试内容还得再看看 修改不利: etc/tests.cmake才是真正决定测试的脚本

This week, you’ll implement the **receiver** part of TCP, responsible for receiving messages from the sender, reassembling the byte stream (including its ending, when that occurs), and determining that messages that should be sent back to the sender for acknowledgment and flow control

acknowledgment: sender要发或重发的next byte的索引
Flow control: sender应该发多少

receive()：
- Set the Initial Sequence Number if necessary.
- Push any data to the Reassembler: Remember that the Reassembler expects stream indexes starting at zero; you will have to unwrap the seqnos to produce these.

测试框架还是蛮有意思的可以学习，以后利用起来

笔记也要有结构

传过来的Wrap32而Reassembler要的是absolute seqence number

string的release 函数有意思啊 压根没这个函数

syn message 也是会带payload的

receive不需要调用send

**还要处理error RST flag的情况: sender发来RST我就不需要有其他操作吗？**

merge
```bash
# 得sudo不然会出错
sudo git merge origin/check3-startercode
```

SANITIZING_FLAGS有什么用

# Lab Checkpoint 3: the TCP sender[S:2024/03/11 45min]

Two party participate in the TCP connection, and each party is a peer of the other. Each peer acts as both “sender” (of its own outgoing byte-stream) and “receiver” (of an incoming byte-stream) at the same time.

the “sender” part of TCP, responsible for reading from a ByteStream (created and written to by some sender-side application), and turning the stream into a sequence of outgoing TCP segments. 

**TCPSender’s responsibility**
- Keep track of the receiver’s window
- Fill the window when possible, by reading from the ByteStream, creating new TCP segments (including SYN and FIN flags if needed), and sending them. 一直发直到window满了，或者没东西可发
- Keep track of which segments have been sent but not yet acknowledged by the receiver we call these “outstanding” segments
- Re-send outstanding segments if enough time passes since they were sent, and they haven’t been acknowledged yet
    - tick method
    - The value of the RTO will change over time, but the “initial value” stays the same.
    - You’ll implement the retransmission timer:
    - Every time a segment containing data (nonzero length in sequence space) is sent (whether it’s the first time or a retransmission), if the timer is not running, start it running so that it will expire after RTO milliseconds (for the current value of RTO)

“automatic repeat request” (ARQ)

协议还是很有用的，只要你遵守了协议，怎么实现不关我的事

(that is, without all of its sequence numbers being acknowledged)

but we don’t want you to be worrying about hidden test cases trying to trip you up or treating this like a word problem on the SAT.

Here are the rules for what “outstanding for too long” means. These are based on a simplified version of the “real” rules for TCP: RFC 6298, recommendations 5.1 through 5.6

retransmission timeout (RTO)