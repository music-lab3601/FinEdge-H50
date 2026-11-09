> 来源交付文档快照。公开仓库的实际目录、资产身份和验收状态见根目录README及provenance。

# 合并提交包的组件与外部依赖

本说明以当前比赛提交代码 ZIP 中的 `软件部分/` 与 `硬件部分/` 为整体范围。软件目录保存 ED4 TCN 数据处理、训练、复核、成功的 TorchScript 导出与冻结 CPU 推理代码；硬件目录保存 FPGA 前端、ARM 管理、NPU 部署和上位机演示工程。软件单包整理阶段尚未取得硬件组工程的说明已由本说明更新，ARM、独立行情源和 GUI 现已随硬件部分提供。

`provenance/deployment_inventory.json` 保留 2026 年 11 月 1 日软件单包收集阶段的原始审阅快照，其中 `conclusion`、`missing_external_components` 等字段记录的是当时软件单包的来源范围；它们不代表当前合并提交包仍缺 ARM、行情源或 GUI。历史快照保持原值，当前提供情况以本说明和同级硬件目录实际文件为准。

## 已随合并包提供的工程

以下 `H` 表示 ZIP 根目录下的 `硬件部分/01_工程源码/finedge_host_demo/versions/realtime_integration_v1/`，路径均相对本提交包。

- FPGA RTL 与 Vivado 工程：`H/artifacts/integrated_full_20261103_214358/project/`，包括在线建簿、整数预处理与 INT8 编码、历史组窗、共享 DDR 写入和结果回传。其 `rtl/bench/g_queue_scheduler.sv`、`g_predict_control.sv` 和 `wt_history.sv` 对应 32 个 DDR 输入槽、滚动历史与整窗 FIFO 衔接以及 2ms/H50 双期限控制。
- ARM 常驻调度与板端管理：`H/queued_v2/showcase_v5/native/integrated_native.c` 及 `board/integrated_owner.py`，包含模型常驻、逐槽输入映射、任务启动和完成处理。
- 独立 MBO UDP 行情派发与 QPC 调度：`H/queued_v2/showcase_v5/native/fair_source.cpp`、`native/platform.hpp` 及 `sender.py` 等支持代码。
- 上位机 GUI、行情源监控与结果追踪：`H/queued_v2/showcase_v5/runner.py`、`dashboard.py`、`telemetry.py`、`evidence.py`、`web/` 及相关入口和配置文件。
- 已编译混合精度模型：`H/artifacts/queue_campaign_20261031_105509/board_bundle/model.json` 与 `model.raw`。模型记录包含输入布局、尺度与混合精度信息；该部署资产已在包内，不能因为重建转换过程的资料不足而说成没有提供模型。

当前配置、连接及使用路径以 `硬件部分/README.md` 为入口。本次说明修订不改变源码、模型、固件或已有验收记录，也没有新执行训练、推理或板测。

## 仍需另备的环境与输入

商业 ICraft SDK、Vivado/厂家构建工具、板卡镜像、驱动和第三方库安装目录不作为自研源码附带。需使用与现有工程匹配的版本和配置；随包编译模型记录为 Icraft v3.36.1、XIR 3.36.1.0、ONNX opset11，实际运行环境应按硬件说明核对。

完整原始行情、训练/验证缓存和上位机显示数据未随本源码包全部附带。`H/queued_v2/showcase_v5/config.json` 中以 `__EXTERNAL_INPUTS_REQUIRED__` 标明待另备的行情与显示输入，并保留部分原 F 盘工程路径；运行前应准备相应真实输入并调整路径。源码和编译模型齐备于各自已归档范围，不表示解压后即可在任意新机直接复现整套演示。

## 尚未完整随包的转换与实测追溯资料

- ED4 TorchScript 至 opset11 ONNX 的完整导出入口/参数、ICraft 编译及混合精度校准过程源码与记录，以及对应 ONNX 文件尚未完整纳入当前包。现有 `model.json`、`model.raw`、输入尺度和 RTL 编码参数属于已经提供的部署资产。
- 量化配对 1,024 窗的完整端点身份清单、实际校准来源以及从冻结权重到 ONNX/ICraft 工件的完整身份链未随包齐备，不能用 RAW1024 的 1,024 条 TRAIN 记录及少量参考窗替代这组评价证据。
- 报告中五分钟运行的完整逐窗日志、统计 CSV 和后续最终验收目录未全部随包。包内保留的旧运行摘要、构建完成但待板验快照和源区间 manifest 各有其历史范围，不能互相替代或将历史状态直接改写为后续验收成功。

10 月 21 日历史交接工件只对应当时转换参考；当前 32 槽、FPGA 前端与混合部署工程应阅读硬件部分的上述实际路径。当前说明准确列出已提供的工程及仍需补足的复现前提，不根据报告叙述补写示例冒充原源码。
