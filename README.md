# FinEdge H50

面向逐笔订单事件的FPGA前端与板载NPU协同预测工程。

```text
订单事件派发 → PL千兆网 → FPGA建簿 → 整数INT8预处理
→ 100条滑窗 → 整窗FIFO → 32槽共享DDR → NPU → FPGA结果UDP
```

## 当前准备阶段

本分支处于公开源码整理和新检出检查阶段。已收集H50硬件、训练、CPU参考、模型转换和板端运行源码。内部保留必要的历史相对层级，源码对应关系见`provenance/source_map.json`。

完整应用的路径迁移、模型重新编译、厂商工程重建和板测由后续验收阶段分别记录。`@HARDWARE_ROOT@`等标记为显式配置模板，渲染流程仍待接入新的目录。来源脚本中的历史阶段门槛属于原实现上下文。

## 目录

| 目录 | 内容 |
|---|---|
| hardware | 当前H50 RTL、ROM、约束、主机和板端程序，以及必要的相对源码层级 |
| software | 模型训练、数据处理、评估和对应支持源码 |
| deployment | ONNX导出、图变换、量化与Icraft编译来源代码 |
| models/cpu | 清理调试元数据后的冻结模型、训练检查点、归一化、运行接口和合成样本 |
| models/npu | 已验收JSON、RAW、输入布局、尺度及归一化 |
| models/onnx | 导出模型与图变换模型，已清理导出栈文档路径 |
| third_party | 附加许可证；以太网组件保留在hardware内的原相对位置 |
| external | 逐文件外部依赖、身份及处理状态 |
| provenance | 来源映射和Git交付清单 |
| tools | 新检出审计与公开模型检查 |

## 新检出检查

使用Python3.12和Git。

```shell
git clone <repository-url> finedge
cd finedge
python -B tools/verify_checkout.py
```

检查覆盖Git实际收录、文件哈希、关键源码/模型/ROM/许可证、5份时序Tcl、XPR输入归属、敏感信息和Windows电脑路径。报告只输出命中位置和规则名称。

## 冻结CPU模型检查

安装NumPy2.2.6与PyTorch2.7.1后执行。

```shell
python -B tools/check_cpu_model.py
```

样本由种子20261109生成，用于接口和数值回归。公开模型的来源、清理范围和哈希见`models/cpu/package_manifest.json`及`models/identity.json`。

## 依赖与复现阶段

FPGA工具为Vivado2018.3、匹配Procise及JFM器件补丁，NPU编译对应Icraft3.36.1。精确外部文件逐项列在`external/dependencies.json`。官方包到当前修改后集成工程的重建仍待验证。

真实行情按提供方条款独立获取。合成接口样本、市场性能实验及模型开发集分别记录，实验指标按运行批次关联。

## 开源范围

自研代码使用MIT。第三方许可证见`THIRD_PARTY_NOTICES.md`。当前为本地准备版本，发布验收状态见`docs/release_status.json`。
