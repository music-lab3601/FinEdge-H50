# 自动检查

每次推送、创建或更新 Pull Request，GitHub Actions 会运行 `.github/workflows/ci.yml`。也可以在 Actions 页面手动触发 `Source checks`。两个任务分别保存日志和结果 JSON，保留 14 天。命令失败、超时或 RTL 成功标记缺失都会使相应任务失败。

## Windows 软件与模型

任务使用 Windows 2025、Python 3.12、PyTorch 2.7.1 CPU 版和 `requirements.txt` 中的依赖，运行统一入口。

```shell
python -B finedge.py check ci
```

本地执行需要 Git、CMake 和 C++17 编译器，CMake 应位于 PATH。命令建立独立配置与新输出目录 `.local/ci/<运行时间>/`，其中包含以下检查。

| 检查 | 验收内容 |
|---|---|
| 依赖与收录 | pip check、Git 文件清单、哈希、许可、模型身份、敏感信息和本机路径规则 |
| 配置与源码 | 57 份模板渲染、Python 语法与内部导入目标 |
| 单元测试 | 执行完整测试集，结果保留跳过数量及测试原始日志 |
| CPU 功能 | 32 个冻结合成输入、UDP 在线建簿和 160 条盘口的 61 窗推理 |
| 公开输入 | 生成并核验合成订单、盘口、时间范围和启动向量，组装板端运行包 |
| 模型导出 | 32 窗 TorchScript/ONNX 一致性、96 窗因果图变换一致性 |
| 定点 ROM | 重新生成四份阈值 ROM 与参数文件，逐字节核对冻结版本 |
| 原生行情源 | 从 C++ 源码编译，发送 180 条 UDP 事件，逐包核对身份和原始字节 |

各子命令的完整输出位于 `logs/`，汇总结果为 `result.json`。输入全部使用明确标记的合成记录。历史 SDK 与真实数据收据相关测试在原始测试日志中保留跳过原因。桌面服务的静态资源、认证 API 和空闲状态由测试集检查。

## Linux RTL 仿真

任务使用 Ubuntu 24.04、Python 3.12 和发行版提供的 Icarus Verilog。先用固定版本及 SHA256 的 actionlint 检查工作流，再安装仿真器并运行以下命令。

```shell
python -B finedge.py simulate --simulator iverilog
```

四个测试台覆盖异步 FIFO 的顺序与背压、整窗 DDR 传送和可见性、环形组窗与段切换，以及最早响应时序。编译与运行退出码、各自成功标记同时作为通过条件。日志与 `result.json` 位于 `.local/simulation/<运行时间>/`。

Vivado XSim 仍通过原有命令运行。

```shell
python -B finedge.py simulate
```

## 构建与真机记录

完整 Icraft 编译使用匹配 SDK 和授权行情，FPGA 全构建使用 Vivado、Procise、JFM 及匹配厂商依赖。它们的已执行结果分别保存在 `int8_rebuild_validation.json` 和 `fpga_build_validation.json`。真机运行记录通过 `evidence_index.json` 绑定固件及输入身份。新构建 BOOT 的板测按首版发布范围延期。

本轮工作流和本地等效执行结果登记于 `ci_validation.json`。首次 GitHub 托管执行在推送后登记运行链接和提交身份。
