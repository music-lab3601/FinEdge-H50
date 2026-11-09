# H50硬件

RTL按网络、建簿、预处理、窗口、内存、调度和结果模块分组。ROM位于`rom/`，约束位于`constraints/`。

`python finedge.py fpga --check`核对外部依赖，`fpga --prepare-only`生成独立工程副本，`simulate`运行四项合成RTL测试。完整构建使用Vivado2018.3及匹配的厂商输入。详细步骤见根目录README。
