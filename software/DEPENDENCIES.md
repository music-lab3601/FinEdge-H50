> 来源交付文档快照。公开仓库的实际目录、资产身份和验收状态见根目录README及provenance。

# 依赖与外部输入

冻结CPU推理的原环境记录：Python 3.12.14、NumPy 2.2.6、Torch 2.7.1+cu128；版本原文保存在冻结package/environment.json和requirements_snapshot.txt，未在本次重新安装或验证。

原训练及核验代码主要使用NumPy/PyTorch；Databento DBN/MBO转换另需databento_dbn，上游采集支持代码可使用requests/tqdm等。标准库无需另装；准确额外依赖以各入口实际imports及历史环境为准，不虚构未保存的精确版本。

真实复训还需原始352B盘口记录、P1/P2筛选与缓存、manifest/路径和新独立执行安排。完整行情与缓存未随源码附带。

当前合并提交包的硬件部分已包含板端ARM常驻C/Python管理、32槽FPGA工程、独立MBO UDP/QPC行情源、GUI及已编译混合精度模型；这些自研工程不是仍待补交的外部依赖。具体位置见EXTERNAL_COMPONENTS.md和硬件部分/README.md，硬件Python依赖另见硬件部分/03_检查与配置/requirements.txt。

上板仍需匹配的开发板、ICraft SDK、系统镜像、驱动和构建工具。随包model.json记录的编译版本为Icraft v3.36.1、XIR 3.36.1.0及ONNX opset11，这些是编译工件内的版本记录，不代表本次已安装或验证全部环境。完整ONNX导出/ICraft编译校准过程源码与记录、量化评价窗的跨格式追溯资料仍未完整随包；已有编译模型与RTL中保存的尺度/常数不应据此被解释为缺失。当前演示配置还引用原F盘行情和显示数据，使用前需按硬件说明准备实际输入并调整路径。
