# 软件依赖

## 无板卡运行与测试

使用64位Python 3.12，先创建独立venv。安装步骤见根目录README，六项直接依赖固定在`requirements.txt`。

| 依赖 | 版本 | 用途 |
|---|---|---|
| NumPy | 2.2.6 | 价量张量、预处理和数值核对 |
| PyTorch CPU | 2.7.1 | 冻结TorchScript推理和模型结构测试 |
| ONNX | 1.17.0 | 模型导出与计算图检查 |
| ONNX Runtime | 1.22.1 | 导出模型输出对照 |
| pyserial | 3.5 | 串口管理接口及其测试 |
| databento-dbn | 0.71.0 | DBN状态消息、行情记录解码及合成数据测试 |

PyTorch通过官方CPU索引安装，实际包版本为`2.7.1+cpu`。安装其余依赖后执行`python -m pip check`，核对依赖完整性。标准库随Python提供，Git作为独立命令行工具安装。

## 训练与板卡构建

正式训练需要配置相应CUDA版PyTorch、训练数据、注册文件和预算记录。板卡构建需要匹配的Vivado、Procise、JFM和Icraft工具链，C++工具需要CMake及编译器。各项通过集中配置选择，外部资产身份见`external/dependencies.json`。

## 历史环境记录

来源资料中保留的Python、CUDA和依赖版本用于说明原实验环境。本轮独立环境的实际安装版本、依赖来源和验证范围由验收记录单独保存。
