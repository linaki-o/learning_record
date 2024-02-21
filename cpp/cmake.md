
先构建主干，再填充枝叶

# 编写CMake的最底层逻辑

最底层逻辑：需要哪些 以及 在哪
头文件的作用是写声明 而声明的作用就是在编译时让编译器知道我有这个函数 放心干

优化：
- 提升效率(程序运行时相关)
- 减少重复
- 减轻依赖

头文件的作用就是减少重复

# 静态库与动态库

在windows的MSVC中，即使你链接的是动态库，也会生成.lib文件，而你需要在文件中链接的是.lib文件，.dll文件会在装入时链接 要把.dll放到.exe文件的目录下

MSVC中生成动态库时还需要在想导出的函数前加关键字

target_compile_definitions() 可以加宏

# CMake大一统

CMake -> CMakeLists.txt -> Makefile -> .a

CMake其实就是一个跨平台的工具 就是一个翻译罢了 各种构建还是得交给平台上的工具

统一的生成可执行文件的命令：cmake --build . 

```bash
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON # 用来输出Makefile的操作
```
# 再探静态动态链接

```bash
-Wl,-rpath,/caser/lib # 这里是添加运行时库搜索路径，告诉运行时编译器在这个路径中查找动态链接库
```

```cmake
// 这两种方法的底层翻译是有区别的
link_libraries()

link_directories()
target_link_libraries()
```

linux下动态链接的不同
- 不需要像MSCV一样加宏定义
- 可以给可执行文件加动态链接地址，当运行时它就会去那个路径找，而不是像windows系统中的动态库加载顺序