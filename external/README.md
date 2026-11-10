# 外部依赖与获取方式

`dependencies.json` 逐项记录目标 URI、字节数、SHA256 和原始交付路径。当前共 766 项登记，其中 FPGA 与启动组件为 756 项。模型 RAW 同时通过 `repo://` 指向 Git 内的冻结工件。

## 可直接生成的公开输入

运行 `finedge.py example generate` 可生成合成 MBO、显示盘口、时间范围和四份启动向量。生成器和 Icraft Host 数值参考均进入 Git，步骤见 `docs/PUBLIC_REPRODUCTION.md`。

原清单中 `data://startup/` 的四个哈希对应历史演示输入。公开合成示例使用自己的 `manifest.json` 和启动参考契约，分别记录来源。

## 厂商工具与 FPGA 集成依赖

| 类别 | 版本或身份 | 获取与配置 |
|---|---|---|
| Icraft | 主机 3.36.1，板端使用匹配 ARM64 运行组件 | 使用开发板配套软件资料，配置 `--icraft` |
| Vivado | 2018.3 build 2405991 | 从厂商软件分发渠道取得，配置 `--vivado` |
| Procise | 2025.1.1 temp | 使用匹配器件数据库与工具许可，配置 `--procise` |
| JFM | 分发标签 5.3.2.1_100AI_LV，脚本实际报告 5.3.1.8 | 使用配套补丁目录，配置 `--jfm` |
| FPGA 输入 | 753 项，含 AI_Mate.edif、24 份 IP DCP、顶层、约束及补丁 | 按 `vendor://fpga/` 布置并逐项核验 |
| BOOT 输入 | FSBL、BL31、U-Boot 共 3 项 | 按 `vendor://boot/` 布置并逐项核验 |

756 项输入对应本项目已验证的匹配集成快照。其中包括厂商基底、生成的 IP 输出和集成版本的顶层及约束。项目自研功能模块位于 `hardware/rtl/`，厂商原工程相关文件按原组件条款配套提供。获取资料时应核对完整清单中的版本、路径和哈希。

已有本项目原修订交付解压目录时，可以自动导入。`--import-source` 指向包含 `versions/` 和 `hardware/` 的原始 `finedge_host_demo` 根目录。

```shell
python -B finedge.py configure --vendor-root <licensed_dependency_cache>
python -B finedge.py fpga --import-source <extracted_original_delivery_root>
python -B finedge.py fpga --check
```

直接取得匹配文件时，按目标 URI 放入厂商缓存。最终检查结果应为 `checked=756`，`missing=[]`、`changed=[]`。源码构建及 BOOT 生成步骤见 `docs/FPGA_BUILD.md`。

## 行情数据

历史研究使用 Nasdaq 原生 ITCH 与 Databento XNAS.ITCH MBO 两类来源。数据提供方的字段定义、使用和再分发条款随相应账户或数据产品适用。[Nasdaq ITCH 资料](https://www.nasdaqtrader.com/Trader.aspx?id=ITCH) · [Databento 文档](https://databento.com/docs)

部署入口使用按原序保存的 MBO56 记录，具体字段位于 `contracts/market/finedge_contract.py`。DBN 文件先由相应解码器读取，再按消息身份、股票、时间范围与交付规则提取。`software/data/` 保存解码、建簿和审计实现。原生 ITCH 使用对应 ITCH 适配路径。

固定模型重编译使用 `data://rebuild/raw_mbo56.bin` 的完整 SHA256 和 `contracts/model/rebuild_profile.json` 中的窗口选择。历史五分钟原运行也引用该完整源。配套演示包提供保留建簿前缀的裁剪副本，两者的哈希和逐字节前缀关系登记于 `docs/evidence_index.json`。

原训练数据与注册收据由训练交付方按数据条款提供。取得数据后，依据研究协议创建独立注册文件，再执行 `finedge.py train`。公开合成输入可用于检查软件与协议流程。

## 公开发布文件

自研源码、允许分发的模型工件、第三方开源组件及原许可、合成输入生成器和验证记录进入 Git。厂商 SDK、网表、IP 检查点、启动组件和真实行情通过各自匹配许可渠道获取。`THIRD_PARTY_NOTICES.md` 说明组件归属，`dependencies.json` 提供可核对的文件身份。
