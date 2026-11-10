# FPGA 完整源构建

## 构建范围与输入

H50 工程的顶层为 `ai7030_edif_top`，兼容工具器件映射为 `xc7z030ffg676-2`，实际目标板为悟净 FMQL30TAI 完整版。自研行情解析、建簿、定点归一化、INT8 组窗、32 槽调度、DDR 传输及结果输出逻辑从当前 Git 源码重新综合、布局布线。

厂商 AI 网表、PS 外设、DDR 和视频相关 IP 通过 `external/dependencies.json` 登记。构建需要 753 项 FPGA 输入及 3 项启动组件，逐项核对 SHA256。其中包含厂商 `AI_Mate.edif` 和 24 份 IP 检查点。构建入口另外生成本项目时钟 IP 的 RTL，并将其并入顶层综合。

## 工具和资料准备

本机验证的工具组合为 Vivado 2018.3 与 Procise 2025.1.1 temp。JFM 分发标签为 `5.3.2.1_100AI_LV`，加载脚本报告的版本在实际构建日志中记录。需要已安装匹配的器件数据库、可运行的厂商补丁及可用工具许可。Python 环境按根目录 README 准备。

```shell
python finedge.py configure --vivado <vivado_executable> --procise <procise_directory> --jfm <jfm_directory>
python finedge.py configure --vendor-root <licensed_vendor_cache>
```

准备厂商缓存有两种方式。已有本工程原始修订交付解压目录时，使用下面的导入命令。`--import-source` 指向含有 `versions` 与 `hardware` 的原始 `finedge_host_demo` 根目录。入口按清单中的 `original_target` 定位文件，在复制前校验所有输入，再复制到配置的厂商缓存中。

```shell
python finedge.py fpga --import-source <extracted_original_delivery_root>
python finedge.py fpga --check
```

自行取得匹配官方资料时，也可按清单的 `vendor://fpga/` 和 `vendor://boot/` 相对层级放入厂商缓存。检查命令应返回 `checked=756`，且 `missing`、`changed` 均为空。具体的厂商文件许可范围见 `THIRD_PARTY_NOTICES.md`。

## 执行构建

工作目录使用新的、较短的纯英文路径，路径中不含空格。Vivado 2018.3 的部分 Windows 工具受路径长度限制，建议选择靠近盘符根的工作目录。

```shell
python finedge.py fpga --prepare-only --work-dir <new_short_ascii_preparation_directory>
python finedge.py fpga --work-dir <another_new_short_ascii_build_directory> --package-boot
```

准备命令仅生成独立工程副本。完整命令会重新准备另一个新目录，并依次完成综合、布局布线、位流、物理审计和 BOOT 打包。相同工作目录只执行一次，失败日志保存在原目录。

工程副本包含以下层级。厂商脚本根据运行目录向上推导工程根目录，因此这些层级是构建接口的一部分。

```text
build/
  hardware/                       本次自研源码、ROM、约束及构建脚本
  third_party/verilog-ethernet/    以太网依赖
  project/
    fpai_demo_vivado.xpr
    rtl/                          厂商 RTL、IP 和 AI 网表
    PS_AI IP/                     厂商 PS 器件仓库
    ip_patch/                     厂商补丁与构建钩子
    fpai_demo_vivado.srcs/         厂商 block design
    fpai_demo_vivado.runs/         本次综合、实现及位流
  physical_audit/                  路由、时序、时钟和例外检查
  boot/                           启动组件、BOOT.bin 与发布收据
```

Vivado 重置综合运行时会移除部分 IP 输出。构建入口在生成运行脚本后，从已核验缓存恢复缺失的厂商输出，并再次核验 24 份 IP 检查点。自研顶层的综合与布线结果保存在新工作目录。

## 验收和产物

`build.log` 记录主流程，两个运行目录内的 `direct_console.txt`、`direct_run_result.json` 记录实际子进程、返回码、完成标记和耗时。`market_synth.dcp` 与 `market_routed.dcp` 分别为本次综合和布线检查点。利用率、功耗、DRC、CDC、IO 和完整时序报告保存在工作目录根。

物理审计要求普通建立与保持时序均通过、路由错误为零、DRC 无 Error，活跃寄存器时钟和最大延迟约束覆盖完整，建簿时钟为 125 MHz。厂商已确认的 AI 内部两项 1 GHz 最小周期模型例外逐引脚检查，其余最小周期和脉宽检查正常执行。两项例外及其数值写入 `boot/release.json`。

命令与完成信息的两个小型异步 FIFO 使用双级 Gray 指针同步后读取 RAM。其叶级 RAM 时钟输出到读寄存器的数据发布路径按 5 ns 限制，异步保持关系仅对该路径豁免。命令 FIFO 的 200 MHz 读域提供至少 10 ns 的发布等待，完成 FIFO 的 125 MHz 读域提供至少 16 ns。读地址路径单独约束。物理审计逐 FIFO 核验实际路径、5 ns 要求、裕量和保持例外，将命中端点写入 `physical_audit/fifo_publication.tsv`。

BOOT 使用厂商后处理生成的 `ai7030_edif_top_disable_icap.bit`，结合已登记的 FSBL、BL31 和 U-Boot 打包。主要发布产物为 `boot/BOOT.bin`、`boot/BOOT.bin.gz` 与 `boot/release.json`。本轮功耗报告采用环境温度25℃、典型工艺及工具默认活动率，属于工具估计。

构建成功时，BOOT 收据状态为 `packaged_pending_board_validation`。该产物已完成DDR、行情接收、NPU结果及演示完整链路的后续真机验收，记录见`docs/board_validation.json`。

## 2026-11-10 构建验收

最终脚本在全新克隆中以一条命令完成综合、布局布线、厂商位流后处理、物理审计及 BOOT 打包。自研输入文件与暂存清单的哈希全部一致。详细结果及产物哈希见 `docs/fpga_build_validation.json`。

普通建立裕量为 0.030 ns，保持裕量为 0.014 ns。TNS、THS 和路由错误均为零，活跃端点的时钟与最大延迟约束覆盖完整。厂家两项 AI 最小周期模型例外及原始 DRC 兼容设置逐项记录。

| 项目 | 使用量 / 可用量 | 占用率 |
|---|---|---|
| LUT | 55853 / 78600 | 71.06% |
| 寄存器 | 89512 / 157200 | 56.94% |
| 布局占用的 Slice | 19639 / 19650 | 99.94% |
| BRAM 等效块 | 216 / 265 | 81.51% |
| DSP | 18 / 400 | 4.50% |

工具估计片上功耗为 3.842 W，使用默认活动率，置信度为 Low，PS7 活动参数未指定。构建收据的BOOT状态为`packaged_pending_board_validation`。同一固件后续真机验收结果独立保存于`docs/board_validation.json`，通过BOOT哈希关联。
