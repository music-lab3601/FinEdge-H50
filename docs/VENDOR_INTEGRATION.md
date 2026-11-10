# 厂商基底与本项目集成的重建

## 文件组成

`hardware/rtl/`保存行情解析、建簿、整数归一化、滑窗、DDR队列及结果通路。`hardware/project/reference.xpr`将这些模块与厂商PS、DDR和NPU接口连接。厂商顶层、PS外设包装及约束还需要应用本项目的集成改动。

`hardware/integration/vendor_edits.json`保存8个文件的重建配方。每项记录原文件字节数与SHA256、按字节偏移执行的编辑，以及最终文件字节数与SHA256。顶层先按配方将换行转换为CRLF，随后执行编辑，其余文件保留原换行。未修改的内容从用户取得的原厂文件读取。补充内容与重建程序随Git交付。

| 基底文件 | 集成内容 |
|---|---|
| rtl/ai7030_edif_top.v | PL网口、User_Reg与User_Ddr连接，启动复位逻辑及前端实例 |
| rtl/ps_ai_peripheral.sv | 关闭视频写DDR通路，固定闲置接口信号 |
| constrs_1/ai7030.xdc | 保留厂商约束，将GP/HP例外限定在旧端点 |
| constrs_1/impl_constraints.xdc | 悟净完整版SDI参考时钟引脚U6/U5 |
| ip_patch/process_control/synthesis_pre.tcl | 匹配已验证JFM版本的调用参数 |
| ip_patch/process_control/implementation_opt_post.tcl | 已验证钩子的精确字节身份 |
| ip_patch/process_control/write_bitstream_pre.tcl | 匹配器件及补丁版本的位流设置 |
| ip_patch/process_control/write_bitstream_post.tcl | 匹配已验证后处理的调用参数 |

前4项以原始`fpai_demo_fpga`为基底，后4项以原始`JFM_Kits`为基底。两套基底的文件身份均写入配方。原厂代码及工具组件按原组件条款获取和使用。原始参考RAR、JFM ZIP和解压目录均属于本机依赖准备资料。

## 执行步骤

取得匹配资料并解压后，选择一个专用厂商缓存目录。传给命令的目录分别包含`rtl/`与`ip_patch/`。

```shell
python -B finedge.py configure --vendor-root <licensed_vendor_cache>
python -B finedge.py fpga --integrate-reference <original_fpai_demo_fpga_directory> --jfm-source <original_JFM_Kits_directory>
```

命令先验证全部基底与目标，再写入8项结果和`integration_receipt.json`。重复执行会核验已生成文件。目标文件内容不一致时会给出具体文件名，便于选择独立缓存处理。

其余依赖包括原厂RTL、AI网表、JFM文件、24个IP检查点及对应生成输出、FSBL、BL31和U-Boot。原始参考压缩包提供IP配置，清单同时记录已验证工具生成的输出身份。仅解压参考压缩包时，仍需使用匹配工具生成或取得这些IP输出。按`external/dependencies.json`的目标位置与哈希准备，再运行检查。

```shell
python -B finedge.py fpga --check
python -B finedge.py fpga --prepare-only --work-dir <new_short_ascii_directory>
```

检查应核对746项，缺失与变化列表均为空。准备阶段仅复制这746项中的FPGA文件，并在独立目录写入工具配置。清单中的8个未使用历史模块与约束、1个旧工具配置和1个旧日志标记为非构建必需。当前工程引用Git中的正式RTL和约束。

## 复核方法

本轮从原厂RAR重新提取基底，从原厂JFM ZIP核对钩子，重建8项集成文件，逐字节对照已完成综合和板测的输入。其余必需文件逐项核对哈希，并在只含清单文件的新目录使用Vivado检查工程引用与综合源文件顺序。自动回归验证基底修改、输出修改、编辑越界、重复执行、目标冲突及后续基底失败时的行为。

公开源码完整性检查同时核对模型、ROM、XPR输入与钩子、Markdown本地链接、Python字面资源路径、HDL包含文件、原生源码及本地头文件。运行方式如下。

```shell
python -B finedge.py check checkout
python -B finedge.py check ci
python -B finedge.py simulate --simulator iverilog
```
