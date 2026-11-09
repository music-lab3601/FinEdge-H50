# H50软件

数据接入与重建位于`data/`，TCN结构位于`model/`，训练与评估位于`training/`，冻结CPU运行和增量窗口位于`runtime/`。

统一命令和依赖安装见根目录README。`python finedge.py cpu check`核验冻结模型，`cpu smoke`运行合成UDP闭环，`train --help`查看注册式训练入口。
