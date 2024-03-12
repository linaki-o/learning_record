
先构建主干，再填充枝叶
# 学习目的
很多开源都是用cmake，需要会看会理解，最后会用

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


ldd命令可以查看可执行文件需要的动态链接库

# 减轻依赖 迈向modern cmake
vscode在linux上默认的构建器是ninja

cmake中有些配置可以直接在vscode中配

更新了一下cmake的版本 通过编译更新的 并且配了一下环境变量

原来不需要ubuntu23版本以上 我WSL就行 只要升级一下cmake版本就好 但是就是不知道为什么一直权限不足 需要我用sudo

include_directories()被废弃了
target_include_directories() modern

不能用一个静态库来归档另一个静态库

PRIVATE 代表只对target生效
PUBLIC 代表还对用到target的生效
INTERFACE 代表target不用 要给别人用

自带头文件的方法

# find_package

find_package会给你定义两个变量：
- OpenCV_INCLUDE_DIRS
- OpenCV_LIBS

find_package会去找其的OpenCVConfig.cmake
会往系统环境变量里找

深度学习部署

# 再搞静态库与动态库
```cmake
# 会生成.o文件
add_library(common_object OBJECT common.cpp)
```
于是乎在其他文件内
```cmake
add_library(add STATIC add.cpp $<TARGET_OBJECTS:common_object>)
```
这样就可以生成可独立使用的静态库了

虽然说这样做可以让每一个静态库都可以单独拎出来用 但是多了好多重复 浪费了好些空间

-fPIC 动态库编译需要的选项

set_target_properties(common_object_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)