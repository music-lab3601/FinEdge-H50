# FinEdge H50

逐笔订单流的FPGA前端与板载NPU协同预测工程。最近100条有效十档盘口构成一个窗口，预测未来第50条有效盘口的中间价方向。三类输出依次为跌、平、涨。

```text
历史订单事件按事件间隔派发
  → PL千兆网 → FPGA解析与建簿 → 整数归一化与INT8编码
  → 100条滑窗 → 整窗FIFO → 32槽共享DDR → 板载NPU
  → FPGA封装结果 → UDP → 上位机界面与日志
```

## 目录

```text
finedge.py                 统一命令入口
config/                    默认参数与研究配置
contracts/                 行情协议、模型输入、量化与发布规范
hardware/
  rtl/                     按network/orderbook/preprocess/window等功能分组
  rom/                     定点阈值ROM与位宽参数
  constraints/             时钟、IO与CDC约束
  ip/                      自研时钟IP配置
  project/                 厂商集成工程的可配置XPR
  integration/             厂商顶层、约束及工具钩子的精确重建配方
  scripts/                 构建、依赖检查与BOOT打包
software/
  data/                    ITCH/DBN接入、建簿、样本准备
  model/                   TCN结构
  training/                特征、训练与评估
  runtime/                 冻结CPU模型、增量预处理与窗口接口
deployment/
  export/                  ONNX导出与因果图变换
  icraft/{base,mixed,native}/  编译阶段及C++辅助工具
  quantization/            定点阈值ROM生成
board/{runtime,native,management}/  板端所有者、原生调用与串口管理
host/{market_source,dashboard,telemetry}/  行情派发、界面与统计
models/{cpu,onnx,npu}/     冻结模型工件
models/normalization.json  唯一冻结归一化参数
shared/                    配置、路径、文件IO与资源队列
tests/                     软件、主机、编译编码及RTL测试
experiments/               历史研究辅助源码与分析工具
third_party/               以太网RTL与Lightweight Charts原许可
external/                  外部文件的获取说明、目标位置与哈希
provenance/                来源、目录映射与发布文件清单
.local/                    本机配置、数据与生成物，由Git忽略
```

正式入口统一使用仓库根目录的`finedge.py`。`experiments/reference/`保留历史辅助实现及其来源身份。当前可执行路径与验证范围见下文。

训练通过`finedge.py train prepare/admit/run`执行，使用明确的数据与缓存清单，输出独立的新权重和记录。输入格式、CPU/CUDA命令及合成训练验收见[训练复现](docs/TRAINING.md)。

当前开源版本已完成源码、完整构建与新固件真机闭环验收。实验与固件对应关系见[证据索引说明](docs/EVIDENCE.md)。

v1.0.1补齐厂商顶层、约束和工具钩子的集成重建配方，并增加源码引用检查。本轮独立克隆执行696项测试，670项通过，26项按依赖和CUDA条件跳过。结果见[交付完整性复核](docs/COMPLETENESS_AUDIT.md)。

v1.0.2进一步接通独立训练复现入口。702项测试中676项通过、26项按条件跳过，实际完成合成小样本缓存准备、训练和检查点重载，另验证8类文件缺失都能被检查发现。结果见[第二轮验收记录](docs/completeness_validation_v2.json)。

## 无板卡验证

使用64位Python 3.12和Git。Windows PowerShell先创建独立环境，再通过该环境的解释器安装依赖与运行检查。

```powershell
git clone https://github.com/music-lab3601/FinEdge-H50.git
cd FinEdge-H50
py -3.12 -m venv .local/python_env
$finedgePython = (Resolve-Path .local/python_env/Scripts/python.exe).Path
& $finedgePython -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& $finedgePython -m pip install -r requirements.txt
& $finedgePython -m pip check
& $finedgePython -B finedge.py check checkout
& $finedgePython -B finedge.py configure --render
& $finedgePython -B finedge.py check imports
& $finedgePython -B finedge.py cpu check
& $finedgePython -B finedge.py cpu smoke --records 160
& $finedgePython -B finedge.py check tests
```

完整Git收录校验通过上述克隆工程执行。Release源码ZIP保存同一提交的文件快照。

后续示例中的`python`同样使用这个环境的解释器。`requirements.txt`包含行情DBN解码库，完整测试会验证其状态消息解析。

`cpu check`验证随仓库提供的32个人工合成窗口。`cpu smoke`独立生成MBO事件，通过本机UDP发送，接收方建簿、增量预处理、组窗并调用冻结CPU模型。160条有效盘口产生61窗，每窗同时对照批量预处理结果。输出保存在`.local/checks/`。此命令的输入和后端均在日志中标明。

`check tests`会显示需要额外历史SDK记录或真实数据的跳过项。缺失项及其原因作为验收结果保留。

## 公开合成输入

```shell
python -B finedge.py example generate --directory .local/public_example
python -B finedge.py example verify --profile .local/public_example/manifest.json
python -B finedge.py example cpu --profile .local/public_example/manifest.json --directory .local/example_cpu --records 160 --speed 10
python -B finedge.py example bundle --profile .local/public_example/manifest.json --directory .local/example_arm_bundle
```

生成 3,070 条人工订单、3,050 条盘口、1/2/5分钟范围和四份启动向量。CPU 示例经 UDP 在线建簿、组窗和推理，160 条盘口产生61窗。合成数据、CPU后端和Icraft Host启动参考均明确记录身份。公开v2合成输入已完成真机闭环验收。完整步骤见[公开复现说明](docs/PUBLIC_REPRODUCTION.md)，外部组件取得方式见[依赖说明](external/README.md)。

## 配置

`config/default.json`保存公开默认值。`configure`保存覆盖项到`.local/config.json`，可重复执行。通过环境变量`FINEDGE_CONFIG`可选择另一份覆盖文件。

```shell
python -B finedge.py configure --data-root <data_directory> --vendor-root <vendor_directory>
python -B finedge.py configure --icraft <sdk_directory> --vivado <vivado_executable>
python -B finedge.py configure --procise <procise_directory> --jfm <jfm_directory> --cmake <cmake_executable>
python -B finedge.py configure --serial-port <serial_device> --host-ip <host_ip> --device-ip <board_ip>
python -B finedge.py check config
```

路径引用采用`repo://`、`work://`、`data://`、`vendor://`和`tool://`五种前缀。57份原模板的路径解析结果写入`.local/rendered/result.json`。模板解析与外部文件是否齐备分别记录，依赖缺失时可从预检报告定位。

## 界面与板卡演示

```shell
python -B finedge.py native
python -B finedge.py demo
python -B finedge.py demo --review <saved_run_directory>
```

空闲界面可以直接打开。运行真实行情需按`external/dependencies.json`准备原始MBO、显示参考、片段范围、四份启动向量及对应固件，并配置`--replay-raw`、`--replay-rows`、`--replay-ranges`、`--startup`和`--firmware-release`。公开合成输入通过`--demo-profile`选择，回退固件使用`--rollback-firmware-release`指定。终端环境变量`FINEDGE_BOARD_PASSWORD`用于串口登录。管理采用PS串口，行情与结果采用PL网口。资源队列串行化板卡操作，正常软件重启后核验新的boot_id和控制器状态。详细步骤见[运行说明](docs/RUNNING.md)。

## 模型与FPGA

```shell
python -B finedge.py model export
python -B finedge.py model polyphase --label trial01 --preserve-taps
python -B finedge.py model calibration --input <normalized_calibration.npy>
python -B finedge.py model compile --kind int8
python -B finedge.py model rebuild --raw-mbo <raw_mbo56.bin> --work-dir .local/rebuild01
python -B finedge.py fpga --check
python -B finedge.py fpga --prepare-only --work-dir <new_ascii_build_directory>
python -B finedge.py fpga --work-dir <another_new_ascii_build_directory> --package-boot
python -B finedge.py simulate
```

模型编译阶段及所需收据见[模型构建说明](docs/MODEL_BUILD.md)。FPGA采用Vivado 2018.3、匹配的Procise/JFM及悟净完整版参考输入。厂商文件按清单导入并核对哈希。构建入口将源文件和厂商输入复制到独立英文路径工作目录，随后综合、布局布线、物理审计和BOOT打包。导入步骤、工具设置、目录约定及产物见[FPGA构建说明](docs/FPGA_BUILD.md)。

## 当前验收

目录迁移、57份模板解析、CPU网络功能闭环、HTTP界面、行情源编译、ONNX导出和图变换、定点ROM再生成、四项RTL仿真已进行本地验证。精确结果以`docs/migration_validation.json`为准。

完整Icraft原生INT8部署重编译已通过本机验收，JSON/RAW与冻结工件逐字节一致，1024窗新旧Host输出逐位一致。结果见`docs/int8_rebuild_validation.json`。H50厂商集成工程已在全新克隆中通过单次完整构建、物理审计及BOOT打包，结果见`docs/fpga_build_validation.json`。本轮新固件已通过AAPL五分钟回放和公开合成输入板测，详见[真机验收](docs/BOARD_VALIDATION.md)。公开合成输入已在新克隆中验证，677项测试中651项通过、26项历史收据缺失跳过。CPU的160条盘口产生61窗，原生派发器180条UDP事件的身份和原始字节全部一致，HTTP界面正常启动退出，两组Icraft Host启动输出逐字节一致。结果见[公开复现验收](docs/public_reproduction_validation.json)。全新克隆审计核对Git实际收录、哈希、许可证、敏感信息及本机路径。

## GitHub验收

Windows云端执行686项测试，660项通过、26项历史收据缺失跳过。Ubuntu四项RTL测试通过。GitHub匿名克隆已核对全部源码、模型、ROM、约束与许可证。完整记录见[云端与克隆验收](docs/github_validation.json)，工作流见[GitHub Actions](https://github.com/music-lab3601/FinEdge-H50/actions/workflows/ci.yml)。

## 自动检查

每次推送和Pull Request会触发软件与RTL检查。Windows任务运行完整测试、CPU网络闭环、模型导出、ROM再生成及原生行情源验证。Linux任务使用Icarus Verilog运行四个RTL测试台。日志作为Actions工件保留14天。

```shell
python -B finedge.py check ci
python -B finedge.py simulate --simulator iverilog
```

本地运行需准备相应的Python依赖、CMake/C++编译器或Icarus工具。配置隔离、验收范围和结果说明见[CI说明](docs/CI.md)。

## 许可

自研代码采用MIT。以太网组件采用MIT，图表组件采用Apache-2.0。厂商SDK/IP及行情输入的获取与再分发依据各自条款。逐项说明见[许可与来源](THIRD_PARTY_NOTICES.md)。
