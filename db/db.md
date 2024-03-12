# Database Storage 

Task: Disk Manager

Volatile: Random Access Byte-Addressable
Non-Volatile: Sequential Access Block-Addressable

**Random access on non-volatile storage is almost always much slower than sequential access.**

<img src="architecture.png">

Memory Mapped I/O Problems
- Transaction Safety
- I/O Stalls
- Error Handling
- Performance Issues

DBMS (almost) always wants to control things itself and can do a better job than the OS.
- 数据页的粒度更加细致，可以针对不同的数据页进行缓存，不需要将整个文件映射到内存中，从而避免了不必要的内存浪费。
- Flushing dirty pages to disk in the correct order.
- Specialized prefetching.
- Buffer replacement policy.
- Thread/process scheduling.

相比之下，使用 mmap 管理内存并不太适合数据库系统。因为 mmap 是将一个文件或者设备映射到进程的地址空间中，而数据库中的数据不仅仅保存在一个文件中，还包括了索引文件、日志文件等多个文件，需要用多个 mmap 区域来管理，这会增加复杂度。此外，mmap 映射的粒度是页，而 buffer pool 可以更加细致地管理内存，优化缓存策略。并且，由于数据库操作的特殊性（如事务管理等），使用 mmap 会带来更多的额外开销和实现难度。因此，对于数据库系统来说，使用 buffer pool 更为适合。

## File storage

Database -> file

It organizes the files as a collection of pages.

Different DBMSs manage pages in files on disk in 
different ways.
- Heap File Organization
- Tree File Organization
- Sequential / Sorted File Organization (ISAM)
- Hashing File Organization

### Heap File

A heap file is an unordered collection of pages with tuples that are stored in random order

就像数组一样 It is easy to find pages if there is only a single file.

Page Directory


## Page Layout

Page Header

Page Layout
- Tuple-oriented
- Log-structured

## Tupled storage

### slotted pages

The DBMS needs a way to keep track of individual tuples.
Each tuple is assigned a unique record identifier.
- Most common: page_id + offset/slot
- Can also contain file location info. 

Each tuple is prefixed with a header
that contains meta-data about it.
- Visibility info (concurrency control)
- Bit Map for NULL values.

denormalize

缺点
- Fragmentation，删除时会有gap
- Useless Disk I/O，我只要一个你却给我一块
- Random Disk I/O，可能要去20个page里读20个tuple

### Log Structured Merge Trees(LSM) 原理 7

KV 数据库

顺序读写磁盘（不管是SATA还是SSD）快于随机读写主存，而且快至少三个数量级。这说明我们要避免随机读写，最好设计成顺序读写。

Instead of storing tuples in pages, the DBMS only stores log records.

To read a record, the DBMS scans the log backwards and recreates the tuple to find what it needs.

**Build indexes to allow it to jump to locations in the log.**


其实就是存储一个page 一个page里包含了对tuple的操作序列就PUT和DELETE 但是没存储tuple，当读时就查找最新的记录，然后依此recreate tuple

每一个tuple都有唯一标识

**Periodically compact the log**
- Sorted String Tables (SSTables)

The issue with compaction is that the DBMS ends up with write amplification. (It re-writes the same data over and over again.)

## Data Representation
 It is up to the DBMS to know how to interpret those bytes to derive the values for attributes. A data representation scheme is how a DBMS stores the bytes for a value.

Variable Precision Numbers: IEEE-754 standard
- 运算快 因为有硬件支持
- 因为round的原因有时候不太准确
- FLOAT, REAL

Fixed-Point Precision Numbers
- 准确
- 实现操作复杂，每一个厂商有自己的实现方法
- NUMERIC, DECIMAL

Variable-Length Data
- a special “overflow” page and have the tuple contain a reference to that page.
-  They are typically stored with a header that keeps track of the
length of the string to make it easy to jump to the next value
- 可能有校验和

## System Catalogs
In order for the DBMS to be able to decipher the contents of tuples, it maintains an internal catalog to tell it meta-data about the databases. The meta-data will contain information about what tables and columns the databases have along with their types and the orderings of the values. Most DBMSs store their catalog inside of themselves in the format that they use for their tables. They use special code to “bootstrap” these catalog tables.

## book

#  Storage Models & Compression

## Database Workloads

> OLTP: Online Transaction Processing

就是用户登录淘宝的那种普通数据库操作
快 短 简单 写的比读的多 一次读一条tuple

> OLAP: Online Analytical Processing

就是淘宝会根据热度放到首页的那种查询

慢 复杂 通常只读 需要先通过OLTP database获取数据

> HTAP: Hybrid Transaction + Analytical Processing

## Storage Models

> N-Ary Storage Model (NSM)

行存储模型

将一个tuple连续的存在一个page，适合OLTP 适用于查找整个tuple

Not good for scanning large portions of the table and/or a subset of the attributes

> Decomposition Storage Model (DSM)

列存储模型

In the decomposition storage model, the DBMS stores a single attribute (column) for all tuples contiguously
in a block of data. 

适合OLAP

Better query processing and data compression

Slow for point queries, inserts, updates, and deletes because of tuple splitting/stitching

两种使用方式
- 固定长度 偏移量索引
- use embedded tuple ids

> Database Compression

They have to strike a balance between speed vs. compression ratio. Compressing the database reduces DRAM requirements. It may decrease CPU costs during query execution

However, there are key properties of real-world data sets that are amenable to compression:
- Data sets tend to have highly skewed distributions for attribute values (e.g., Zipfian distribution of the
Brown Corpus).
- Data sets tend to have high correlation between attributes of the same tuple (e.g., Zip Code to City, Order Date to Ship Date).

Given this, we want a database compression scheme to have the following properties:
- Must produce fixed-length values. The only exception is var-length data stored in separate pools. This because the DBMS should follow word-alignment and be able to access data using offsets.
- Allow the DBMS to postpone decompression as long as possible during query execution (late materialization).
- Must be a lossless scheme because people do not like losing data. Any kind of lossy compression has to be performed at the application level.

> Compression Granularity

- Block Level
- Tuple level
- Attribute Level
- Columnar Level

> Naive Compression

用通用的压缩算法来压缩

An example of using naive compression is in MySQL InnoDB. The DBMS compresses disk pages, pad them to a power of two KBs and stored them into the buffer pool.

If the goal is to compress the entire table into one giant block, using naive compression schemes would be impossible since the whole table needs to be compressed/decompressed for every access. Therefore, for MySQL, it breaks the table into smaller chunks since the compression scope is limited.

这种压缩不关心数据高层次的含义以及语义 所以不能late materialization, since the DBMS will not be able to tell when it will be able to delay the decompression of data

> Columnar Compression
- Run-Length Encoding (RLE)
- Bit-Packing Encoding
- Mostly Encoding
- Bitmap Encoding
- Delta Encoding
- Incremental Encoding
- Dictionary Compression
    - A dictionary compression scheme needs to support fast encoding/decoding, as well as range queries
    - It is not possible to use hash functions
    - The encoded values also need to support sorting in the same order as original values. This ensures that results returned for compressed queries run on compressed data are consistent with uncompressed queries run on original data. This order-preserving property allows operations to be performed directly on the codes
# Memory Management
> Locks Vs Latches

- Locks: A lock is a higher-level, logical primitive that protects the contents of a database (e.g., tuples, tables, databases) from other transactions. Transactions will hold a lock for its entire duration. Database systems can expose to the user which locks are being held as queries are run. Locks need to be able to rollback changes.
- Latches: A latch is a low-level protection primitive that the DBMS uses for the critical sections in its internal data structures (e.g., hash tables, regions of memory). Latches are held for only the duration of the operation being made. Latches do not need to be able to rollback changes.
> Buffer Pool Meta-data

The buffer pool must maintain certain meta-data in order to be used efficiently and correctly
<img src="buffer_pool.png">

page table != page directory

The page table also maintains additional meta-data per page, a dirty-flag and a pin/reference counter

> Memory Allocation Policies

global policies/local policies

>  Buffer Pool Optimizations
- Multiple Buffer Pools
- Pre-fetching
- Scan Sharing (Synchronized Scans)
- Buffer Pool Bypass
> OS Page Cache

Most DBMS use direct I/O to bypass the OS’s cache in order to avoid redundant copies of pages and having to manage different eviction policies.

> Buffer Replacement Policies
- Least Recently Used (LRU)
- CLOCK
- LRU-K
    - 解决LRU算法“缓存污染”的问题，其核心思想是将“最近使用过1次”的判断标准扩展为“最近使用过K次”。也就是说没有到达K次访问的数据并不会被缓存，这也意味着需要对于缓存数据的访问次数进行计数，并且访问记录不能无限记录，也需要使用替换算法进行替换。
    - K值增大，命中率会更高，但是适应性差（清除一个缓存需要大量的数据访问，一般选择LRU-2）。
- localization per query
    DBMS根据每个事务/查询选择要删除的页面。这最大限度地减少了每个查询对缓冲池的污染。
- background writing

> Other Memory Pools
用来缓存其他东西

# Hash Table
> Data structure

There are two main design decisions to consider when implementing data structures for the DBMS:
- Data organization
- Concurrency

> Hash Table

A hash table implementation is comprised of two parts:
- Hash Function
- Hashing Scheme

> Static Hashing Schemes
This means that if the DBMS runs out of storage space in the hash table, then it has to rebuild a larger hash table from scratch, which is very expensive
- Linear Probe Hashing
    - Non-unique Keys
        - Seperate Linked List
        - Redundant Keys
- Robin Hood Hashing 平均一下每一个元素离自己原本位置的距离
- Cuckoo Hashing 多个表 驱逐其他元素
> Dynamic Hashing Schemes

- Chained Hashing 链表
- Extendible Hashing 溢出就分裂
- Linear Hashing 为未来做打算 指针指向要分裂的bucket
# Trees Indexs
- For table indexes, which may involve queries with range scans
- The DBMS ensures that the contents of the tables and the indexes are always logically in sync
> B+ Tree 106
- 支持线性查找
- 有利于读取一整个块的数据
- O(log(n))
- B-Trees stores keys and values in all nodes, while B+ trees store values only in leaf nodes.
- <img src="b+tree.png">
- 特点
    - M way 有几个子节点 如二叉树为2
    - It is perfectly balanced (i.e., every leaf node is at the same depth).
    - Every inner node other than the root is at least half full (M/2 − 1 <= num of keys <= M − 1).
    - Every inner node with k keys has k+1 non-null children.
- Two approaches for leaf node values are record IDs and tuple data
- Clustered Indexes
    - https://blog.csdn.net/ak913/article/details/8026743
    - 正文内容本身就是一种按照一定规则排列的目录称为"聚集索引"
    - The table is stored in the sort order specified by the primary key
    - 就是B+树的叶子结点和数据在磁盘上存储页的顺序是一样的
    - 要是非聚集的话 索引要访问不同的页 随机读效率很慢 可以把要读的页编号排序 然后读
> B+Tree Design Choices 30
- Node Size
    - 越慢的存储设备 越大 因为一次读写可以读比较多 最好和page size一样 这样就可以一页一页的来 不用读一个结点时这个页读一点那个页读一点
    - 与workload 相关 OLAP因为经常访问leaf 所以越大越好 OLTP经常要查找 小一点可以减少冗余数据
- Merge Threshold
- 可变长度的key
- 结点内部搜索
- 前缀压缩
- 冗余key处理
- Bulk insert
    - 减少了插入过程中重构和读取随机页的开销
# Index Concurrency Control 66
> Protocol
- Logical Correctness
    - 一致性
- Physical Correctness
- 读者写者 当有写者来时就不允许继续添加读者 防止饥饿
> Latch Implementations
- Blocking OS Mutex 
    - std::mutex
- Test-and-Set Spin Latch
    - std::atomic<T>
- Reader-Writer Latches
    - std::shared_mutex

> Hash Table Latching
- 为每一个Bucket一个锁 一个全局锁用在要改变表结构时
- Page Latches
- Slot Latches
- compare-and-swap (CAS)无锁 和spinlock那一句原子操作很像

> B+Tree Latching
- Steps
    1.  Get latch for the parent.
    2. Get latch for the child.
    3. Release latch for the parent if the child is deemed “safe”. A “safe” node is one that will not split,
- 总是要获得root的锁 会成为瓶颈
    - 乐观锁 悲观锁 假设不会改root 当会改root的情况发生时再回滚 先设Read Latch
- 扫描叶子结点时可能会发生"请求与保持情况"造成死锁 解决方法:按顺序加锁
# Sorting & Aggregations Algorithms 51
- 不能假设查询结果都在内存中
- 需要依赖buffer pool
- 最大化sequential I/O
- 早物化就是排序的KV直接包含该tuple的全部数据
- 晚物化就是KV包含的是tuple的record id
> Sorting
- Two-way Merge Sort 有点像双指针 buffer pool中的页数为B 需要排序的页数为N 
    - 时间复杂度 1 + ⌈log2N⌉
    - The total I/O cost is 2N × (# of passes)
- General (K-way) Merge Sort
    - buffer pool中的页数增加了
- Double Buffering Optimization
    - 预取
- Using B+Trees
>  Aggregations
- Sorting
    - When performing sorting aggregations, it is important to order the query operations to maximize efficiency.
- Hashing
    - Partition
        - 分类
    - ReHash
        - 再分类
        - During the ReHash phase, the DBMS can store pairs of the form (GroupByKey→RunningValue) to compute the aggregation.

# Joins Algorithms
- 讲的主要是内联
- 内联是为了The goal of a good database design is to minimize the amount of information repetition
> 内联的输出：
- 早物化：数据
- 晚物化：Records Id
> 性能分析
- 指标为I/O次数
- Nested Loop Join
    - 小表当外表，一般会尽可能把多的外表页放到buffer pool中
    - 小意味着tuple少 或者 页数少
    - Simple Nested Loop Join
        - This is the worst case scenario where the DBMS must do an entire scan of the inner table for each tuple in the outer table without any caching or access locality.
    - Block Nested Loop Join
    - Index Nested Loop Join
- Sort-Merge Join
    - This algorithm is useful if one or both tables are already sorted on join attribute(s) (like with a clustered index) or if the output needs to be sorted on the join key anyways.
    - 当the join attribute有很多重复的时候，性能会退化成Simple Nested Loop Join
- Hash Join 点查询很快
    - Basic Hash Join
        - Build 根据外表来创建一个hash table
        - Probe 对内表使用hash func来找 可以用Bloom Filter（in CPU caches）来优化 防止记录不存在时的无意义I/O---definitely no or probably yes.
    - Grace Hash Join / Partitioned Hash Join (When the tables do not fit on main memory)
        - Build 对两表recursive partitioning 来构建hash table 有时候bucket会很大 所以还得hash
        - 对bucket层面进行nested loop join
    - Hybrid hash join optimization: adapts between basic hash join and Grace hash join;

#  Query Processing I
## Processing Models(树形结构)
> Iterator Model(Volcano or Pipeline model)
- 每次从children里拿一条record然后分析 
- 但是有些可能会堵塞直到children的所有数据输出 Examples of such operators include joins, subqueries, and ordering (ORDER BY). Such operators are known as pipeline breakers
- 很适合LIMIT操作符
- 函数调用的开销可能很大
> Materialization Model
- is a specialization of the iterator model where each operator processes its input all at once and then emits its output all at once.
- 适合OLTP 因为OLTP通常只需要小部分数据
> Vectorization Model
- each operator emits a batch (i.e. vector) of data instead of a single tuple.
- 适合OLAP
- The vectorization model allows operators to more easily use vectorized (SIMD Inter的某个硬件结构) instructions to process batches of tuples.
> Processing Direction
- Top-to-Bottom:Tuples are always passed with function calls
- Bottom-to-Top:Allows for tighter control of caches / registers in operator pipelines

##  Access Methods(how the DBMS accesses the data stored in a table)
> Sequential Scan
- 扫描page扫描tuple
- The DBMS maintains an internal cursor that tracks the last page/slot that it examined
- 优化方法
    - Prefetching
    - Buffer Pool Bypass
    - Parallelization
    - Late Materialization
    - Heap Clustering
    - Approximate Queries (Lossy Data Skipping)
    - Zone Map (Lossless Data Skipping) 先统计表的信息 fetch的时候先看统计信息
> Index Scan(the DBMS picks an index to find the tuples that a query needs)
- 问题：有多个索引时选择哪一个索引 小林Mysql里有写
## Modification Queries(更改表数据时需要把一些相关的元数据给改了)
- Halloween Problem
## Expression Evaluation
- <img src="ev.png">
- Evaluating predicates in this manner is slow because the DBMS must traverse the entire tree and determine the correct action to take for each operator. A better approach is to just evaluate the expression directly (think JIT compilation). Based on a internal cost model, the DBMS would determine whether code generation will be adopted to accelerate a query.
- AOT


# Query Execution II 
- 并发下的查询优化
## 好处
- 提高每秒查询次数即吞吐量 降低时延即每一个查询所需时间
- 提高用户层面的响应性和可用性即可以更快响应业务
- 降低total cost of ownership (TCO)即硬件的维修费用和软件的授权费用 因为效率高了利润就高了
## 并发vs分布式
- 并发结点间数据的传输不是问题 而分布式反之 要在这方面下很大功夫
- 但是这些对于DBMS来说都是透明的 不管是并发还是分布式 生成的结果都需要与单结点数据库是一样的
## Process Models
- 定义：数据库系统在多用户环境下支持并发请求的模型
- worker：负责DBMS工作的某一个组件
> Process per Worker
- <img src="ppwm.png">
- 需要依赖OS调度 需要共享内存来传输数据和保存全局结构（这部分带来很大的开销）一个worker的错误不会造成这个系统的崩溃
> Thread per Worker
- pthread 一个进程多个线程 DBMS全权管理线程  一个线程错误整个进程崩溃
> Embedded DBMS
- <img src="eds.png">
## Inter-Query Parallelism
- 查询语句与查询语句层面的并发 提高吞吐量和减低时延 
- 涉及到事务
## Intra-Query parallelism
- 一个语句中算子层面的并发 减低长时间查询的时延
- 像生产者消费者模型
- <img src="iop.png">
- DBMS可以用多线程处理一份中心数据或者将数据切分成多份来处理
> Intra-Operator Parallelism (Horizontal)
- 将数据分成多份 多线程在多份数据上执行相同操作 最后由一个exchange算子来Gather，Distribute，Repartition
> Inter-Operator Parallelism (Vertical)
- 像pipeline一样 一个线程负责一个算子 一个阶段一个算子
> Bushy Parallelism
- <img src="bp.png">
- 结合体
## I/O Parallelism
- 解决磁盘是瓶颈的问题
- 分割数据到多个设备来提高并发度
> Multi-Disk Parallelism
- RAID 将物理磁盘阵列抽象成逻辑的单个磁盘
    - RAID 0 将数据页分割到不同磁盘 提高数据读写的并发度
    - RAID 1 将数据页备份到不同磁盘 提高数据读的并发度
    - RAID 5 将一个磁盘存储其余磁盘的异或数据 相当于前2个的hybrid
> Database Partitioning
- 将数据库分割到不同的磁盘位置上
- This is easy to do at the file-system level if the DBMS stores each database in a separate directory. The log file of changes made is usually shared
- 也可以将表分割到磁盘的不同位置上
    - 垂直分割 将数据大的不经常访问的列分割到磁盘的其他位置 但是逻辑上数据依然是一块的
    - 水平分割 根据hash, range, or predicate partitioning等partitioning keys把tuple子集给分到不同位置上
# Project Zero
- 语法糖
```cpp
bool is_end_{false};
```

- move constructor
```
On declaring the new object and assigning it with the r-value, firstly a temporary object is created, and then that temporary object is used to assign the values to the object

copy constructors就是拷贝一份，而不是所只是把指针指向堆上的对象
    Move(const Move& source)
move constructors就是移动语义了，就是把对象指针指向那个对象 在合适的情况下可以提高效率
    Object_name(Object_name&& obj)
    切记要Nulling out the pointer to the temporary data


 
```

- noexcept: https://zhuanlan.zhihu.com/p/503963684

- move可以用在container上
- 指向unique_ptr的指针可以避免返回的时候unique_ptr的拷贝复制 或移动复制 导致源unique_ptr失效

> Trie Class
- 有头结点
- 不允许插入重复的key
- pair的second是可以改变的 first不行
> C++11 多线程

```bash
 g++ one.cpp -std=c++11 -pthread
 ```

意味着 thread 不可被拷贝构造

```cpp
std::ref(n);
std::move(t3);
```

std::mutex 的成员函数
- 构造函数，std::mutex不允许拷贝构造，也不允许 move 拷贝，最初产生的 mutex 对象是处于 unlocked 状态的
- lock()
- unlock()
- try_lock()
> 二分

最大值最小化条件
答案在一个固定区间内；
可能查找一个符合条件的值不是很容易，但是要求能比较容易地判断某个值是否是符合条件的；
可行解对于区间满足一定的单调性。换言之，如果x是符合条件的，那么有x+1或者x-1也符合条件。（这样下来就满足了上面提到的单调性）

三分法