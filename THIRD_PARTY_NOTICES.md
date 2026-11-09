# 许可与来源

团队自研代码采用根目录MIT许可证。第三方组件依原许可证分发。

| 组件 | 许可 | 随仓库保留 |
|---|---|---|
| verilog-ethernet及AXI Stream基础组件 | MIT | 源文件许可头、COPYING、AUTHORS、RTL及5份时序Tcl |
| TradingView Lightweight Charts | Apache-2.0 | 原库、LICENSE和NOTICE |
| PyTorch、NumPy、ONNX等运行依赖 | 各自许可 | 依赖安装说明 |

厂商IP、JFM补丁、SDK、器件数据库、BOOT输入和完整位流作为外部组件逐文件登记于`external/dependencies.json`。各组件按厂商条款取得。

模型资产由团队模型导出和编译流程产生。CPU公开副本清理调试路径和检查点元数据中的电脑路径，张量权重保持一致。NPU JSON及RAW保持已验收工件字节。

真实行情及来源于行情的旧自检样本按数据提供方条款取得。当前仓库中的公开CPU测试样本由固定随机种子独立生成，性质为人工合成。
