> 来源交付文档快照。公开仓库的实际目录、资产身份和验收状态见根目录README及provenance。

# ED4 H50 软件源码

本目录整理参赛作品中 ED4 H50 软件研究与交付代码，主版本为 `ED4_feature42_lr0.001_seed42` 第 3 轮。源码按原工程相对目录保存在 `project/`，原源码与模型资产保持原字节，另附冻结推理模型和接口校验样例。具体文件、来源及 SHA256 见 `SOURCE_MANIFEST.json`。当前提交 ZIP 的顶层还包括 `硬件部分/`；本说明中的软件研究目录范围不代表整包只包含本目录。

## 版本身份

- 任务：最近 100 条十档订单簿记录，预测未来第 50 条端点中间价相对当前的跌、平、涨；标签为零阈值，不使用未来均价。
- 特征：40 维逐档卖价、卖量、买价、买量；每条记录的 20 个价格通道相对该条自身中间价变换；数量保持原值；用 TRAIN 唯一记录的冻结均值和总体标准差标准化。FP64 处理后转 FP32，无数量 log1p。
- 模型：32 隐藏通道、5 个因果膨胀残差块、每块两层 kernel 3 卷积，dilation 为 1/2/4/8/16；末时刻表示输出 3 类。权重归一化训练参数 33539，折叠部署参数 33219。
- 固定主模型为 seed42/epoch3，原 argmax，不叠加 ED3 平类偏置或 ED5 小学习率方案。seed43 仅为预定配对复核。
- 训练日为 2025-12-08 至 12，DEV 为 12-15 和 12-16；DEV 6612655 窗，Accuracy 46.8784020942%，Macro-F1 0.463728341443。这是历史开发集结果，本次整理没有重训或新增评价。

## 代码阅读入口

`project/scripts/tcn_existing_data_ed4_features.py` 实现相对价格特征及 TRAIN 统计；`tcn_existing_data_ed4_train.py` 为本版训练入口。

`project/scripts/tcn_research_p5_model.py`、`tcn_candidate.py` 是权重归一化 TCN 与基本结构；`tcn_research_data_p2.py`、`tcn_research_evaluate_p2.py` 提供样本与指标支持。其个别旧任务常量属于历史实现，交付输入规则以冻结推理包的 `package_manifest.json`、`runtime.py` 和 `normalization.json` 为准。

`project/scripts/tcn_existing_data_ed6_train.py`、`tcn_existing_data_ed6_data.py` 及其审计程序对应配对复核。依赖闭包中附带的其它历史支持模块不表示另一个模型是主交付。

成功导出使用 `project/scripts/tcn_existing_data_ed7_compat_export.py`、`tcn_existing_data_ed7_compat_runtime.py` 与相应 contract/audit/test/finalize 文件。原 ED7 导出有已记录的 ScriptModule API 失败历史；若包含旧导出代码，它仅供历史依赖与审查，不替代 compat 成功链。

完整的原冻结推理目录为：

`project/artifacts/tcn_existing_data_optimization_2026-10-17/step07_delivery/torchscript_compatibility_v3/package/`

其中 `runtime.py`、`model_cpu.ts`、`normalization.json`、`package_manifest.json`、`source/` 和 `requirements_snapshot.txt` 必须成套使用。该原包 manifest SHA256 为 `bf8244237ac5fc4726b33fe2aee49a15b6f5ac2fdb46485bbd6289933fbdabee`。`native/best.pt` 是原训练容器备份，不是训练续跑指令；黄金窗仅为接口核验参考。

`project/scripts/build_tcn_raw_handoff_20261020.py`、`package_tcn_raw_handoff_20261020.py`、`verify_tcn_raw_handoff.py` 及相应 raw handoff 参考程序是交硬件同学的原始盘口接口衔接代码。其 1024 条 TRAIN 记录不是技术报告中另一组 1024 个量化评价窗，不混用两者成绩。

## 使用与复现范围

本包是参赛源码归档，不会自动启动训练、下载行情、调用板卡或重开历史执行预算。真实训练依赖原数据、独立工作目录、路径配置及明确的新执行安排；原脚本中的历史角色已结束，不能通过清空标记或修改旧预算来复用。

主推理依赖以原冻结目录的 `requirements_snapshot.txt` 和 `environment.json` 为准；整理程序没有安装或升级环境。采集/数据解析的额外第三方库以对应源码 imports 和随包依赖说明为准。商业 Icraft SDK、板卡镜像、驱动和第三方库安装目录不作为自研源码附带。

本包包含源码、必要配置、冻结推理资产及少量接口黄金样例，不附完整原始行情、训练/验证大缓存、虚拟环境或模型搜索目录。数据来源与历史 source quality=false、flag64、控制覆盖未知和 DEV 反复使用限制均保持原记录，不因源码整理而改变。

当前合并提交包的 `硬件部分/01_工程源码/finedge_host_demo/versions/realtime_integration_v1/` 已包含 FPGA RTL 与 Vivado 工程、32 槽队列和双期限控制、ARM 常驻 C 调度程序、Python 板端管理、独立 MBO UDP/QPC 行情源以及展示 GUI；其中 `queued_v2/showcase_v5/runner.py` 为当前演示入口，`queued_v2/showcase_v5/native/` 保存常驻调度及行情源源码。编译后的混合精度模型位于该目录下 `artifacts/queue_campaign_20261031_105509/board_bundle/model.json` 和 `model.raw`。这些文件由硬件部分提供，不能沿用软件单包整理阶段“ARM、行情源和 GUI 未提供”的旧结论。组件路径、仍需另备的环境/数据，以及尚未完整随包的转换校准与实测追溯资料见 `EXTERNAL_COMPONENTS.md`；当前源码归档不等于换机即可直接复现全部训练与板测。

## 文件核验

`SOURCE_MANIFEST.json` 给出实际打包文件、原路径、大小、SHA256 和用途；`SHA256SUMS.txt` 便于逐项核对。ZIP 在交付前已经进行逐 member 的 SHA256、大小和 CRC 校验，并对 Python 源码做 AST 语法核验；没有执行交付源码、加载模型或重新验证市场准确率。
