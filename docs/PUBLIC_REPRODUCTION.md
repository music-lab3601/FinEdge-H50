# 公开示例与复现路线

首个开源版本采用源码与构建验证范围。已完成的验收包括独立 Python 环境、CPU 网络闭环、完整模型转换、FPGA 源构建及 BOOT 打包。新生成固件的真机验收列为后续验证项。

## 先运行公开合成示例

按根目录 README 安装 Python 3.12 环境及依赖后，在仓库根目录运行以下命令。每个输出目录选择一个新目录。

```shell
python -B finedge.py example generate --directory .local/public_example
python -B finedge.py example verify --profile .local/public_example/manifest.json
python -B finedge.py example cpu --profile .local/public_example/manifest.json --directory .local/example_cpu --records 160 --speed 10
python -B finedge.py example bundle --profile .local/public_example/manifest.json --directory .local/example_arm_bundle
```

生成器创建一个 SYNTH 股票的人工订单流。前缀建立十档买卖盘口，随后按固定算法修改订单价格和数量。交易日历使用固定的 2025-12-15 EST 时间戳，订单内容由代码生成。

| 文件 | 内容 |
|---|---|
| market/raw56.bin | 3,070 条 MBO56 原始订单事件，包含清空、新增与修改 |
| display/rows.bin | 3,050 条独立重建的十档盘口及逐条身份 |
| ranges.json | 1、2、5 分钟范围及 50 条未来盘口参考 |
| startup/golden_input.i8 | 4,800 字节全零原生 INT8 输入 |
| startup/golden_logits.f32 | 该输入对应的三个 Icraft Host 输出 |
| startup/dual_slot_input.i8 | 两个不同输入，共 9,600 字节 |
| startup/dual_slot_logits.f32 | 两个输入对应的六个 Icraft Host 输出 |
| manifest.json | 数据性质、生成算法、冻结模型身份及逐文件哈希 |

1、2、5 分钟的有效盘口数分别为 600、1,200、3,000，对应 501、1,101、2,901 次推理机会。每段包含前 99 条组窗预热，步长为 1。额外 50 条盘口用于方向兑现参考。

`verify` 会校验文件哈希，再从生成算法重新构造订单流并核对盘口。CPU 示例通过 UDP 接收原始事件，在线建簿、逐条归一化、组窗和推理。默认 160 条盘口产生 61 窗，每窗对照冻结批量预处理结果。派发线程按事件间隔独立计时，输出目录保存实际派发时刻、窗口身份、输入张量哈希和 logits。

合成示例用于协议、输入组织和软件功能验证。结果文件明确记录 `data_kind=synthetic`、`backend=CPU` 与模型身份。金融效果评价使用有来源记录的行情与标签。

## 启动向量的来源

启动输入使用全零 INT8 编码和固定的整数序列，按原生布局 `[1,3,1,100,16]` 排布，补位通道为零。对应输出由当前冻结 NPU 模型在 Icraft 3.36.1 HostBackend 中实际计算。逐字节 FP32 参考、模型和工具源码哈希保存在 `contracts/examples/synthetic_startup.json`。

生成这些文件使用 NumPy 和公开的参考契约。SDK 用户可编译 `deployment/icraft/native/cpp` 中的 `d_integer_host`，将两个输入串联后使用 `--mode forward` 再次核验 `logits.f32`。

## 为后续板端运行准备资料

`example bundle` 使用现有板端打包函数，组装冻结模型、四份启动向量、PHY 配置、原生 C 调用代码及 Python 所有者。Python 文件按 3.8 语法检查，`config.json` 记录逐文件身份。输出状态为 `prepared`，真机验证状态为 `deferred`。

需要运行板卡时，显式选择公开输入清单和固件发布记录。

```shell
python -B finedge.py configure --demo-profile .local/public_example/manifest.json
python -B finedge.py configure --firmware-release <firmware_release.json>
python -B finedge.py configure --rollback-firmware-release <previous_firmware_release.json>
python -B finedge.py native
python -B finedge.py demo
```

固件身份读取所选 `release.json` 的 `boot_sha256`。回退记录通过单独的配置项指定。界面显示 SYNTH 和“公开合成订单”，运行清单也保存该数据身份。PS 串口、PL 网口、ARM64 Icraft 运行组件及板端登录配置按 `docs/RUNNING.md` 准备。当前版本的最新 BOOT 上板验收由后续任务完成。

现有原生派发程序使用逻辑核 0、2、4、6，板卡演示需要这些逻辑核可用。

切回原有行情配置时，可在 `.local/config.json` 将 `inputs.demo_profile` 设为 `null`，并选择相应的行情、显示参考、启动向量和固件路径。

## 使用授权行情重编译模型

冻结重编译数据身份、校准窗口和验收窗口选择位于 `contracts/model/rebuild_profile.json`。该入口使用 2025-12-15 AAPL 的固定工程输入，其完整 MBO56 文件大小为 328,580,952 字节，SHA256 登记在 `external/dependencies.json`。

```shell
python -B finedge.py configure --icraft <sdk_directory> --cmake <cmake_executable>
python -B finedge.py model rebuild --raw-mbo <matching_raw_mbo56.bin> --work-dir .local/rebuild01
```

入口从原始 MBO 重建 450,261 条盘口，生成 256 个校准窗口和 1,024 个验收窗口，再执行完整转换与 Host 核验。原始数据由使用者按数据提供方条款独立取得，下载和格式说明见 `external/README.md`。自选日期或片段使用独立的输入身份与实验记录。

## 训练入口与交付边界

训练实现位于 `software/training/`，研究协议位于 `config/research/`，类别、特征和权重身份由模型契约固定。训练复跑需要 2025-12-08 至 12 日的授权数据、登记文件和缓存。相关输入按训练协议外部准备。

当前公开验收覆盖冻结权重的 CPU 推理、ONNX 导出、Icraft 完整重编译和部署构建。各阶段验证记录位于 `docs/`。历史板测数据与新构建产物的对应关系见 `docs/EVIDENCE.md`。

## 本轮验收记录

公开示例已在独立新克隆及Python环境中运行。160条有效盘口产生61窗，原生派发器180条UDP事件逐包核对通过，57份配置模板渲染通过，界面经正常入口启动并关闭。两组启动输入重新调用Icraft Host，六个输出与冻结参考逐字节一致。677项测试中651项通过，26项因历史收据缺失跳过。各项输入身份与结果见[公开复现验收](public_reproduction_validation.json)。
