# 统一运行说明

## 配置与生成物

所有命令从仓库根目录执行。默认输出位于`.local/`。配置覆盖文件、下载数据、SDK、构建结果和运行记录单独保存。本机绝对路径写入`.local/config.json`。

| 引用 | 默认位置 | 用途 |
|---|---|---|
| repo:// | 仓库根目录 | Git内源码和模型 |
| work:// | .local/ | 构建、日志、运行状态 |
| data:// | .local/data/ | 独立取得的行情、校准及启动样本 |
| vendor:// | .local/vendor/ | 厂商参考输入和固件 |
| tool:// | configure指定的安装根目录 | Icraft、Vivado、Procise、JFM、CMake |

`python finedge.py configure --render`可重复执行。它解析原57份模板并写入`.local/rendered/`，保留逐文件结果。正式Python代码通过`shared.paths`动态读取配置。Vivado工程在独立构建目录中物化。历史实验文件中的路径引用仍按各自输入身份解析，所需历史收据由相应测试显式报告。

## CPU功能闭环

```shell
python -B finedge.py cpu check
python -B finedge.py cpu smoke --records 160
```

第一条核验冻结模型与32窗合成参考。第二条通过本机UDP派发人工订单，接收方在线建簿。每条有效盘口按自身中间价变换价格，使用冻结float64均值和标准差归一化，然后转float32进入100条环形窗口。每窗与批量预处理逐元素核对，再运行CPU模型。结果记录在`.local/checks/cpu_network.json`。

## 板卡输入

公开合成示例与板端运行包的离线组装步骤见 `PUBLIC_REPRODUCTION.md`。通过 `configure --demo-profile <manifest.json>` 选择输入清单，界面及运行记录显示 SYNTH 身份。固件与回退固件分别使用 `--firmware-release` 和 `--rollback-firmware-release` 指定，BOOT 身份读取对应发布记录。

外部依赖文件列出每个目标URI、字节数和SHA256。真实演示需准备以下输入。

- 原始56字节MBO记录文件，保留重建盘口所需的订单前缀。
- 同源显示参考`rows.bin`及1/2/5分钟范围清单。
- `golden_input.i8`、`golden_logits.f32`、`dual_slot_input.i8`、`dual_slot_logits.f32`四个启动验证文件。
- 对应固件目录中的`release.json`、`BOOT.bin`和`BOOT.bin.gz`。

```shell
python finedge.py configure --replay-raw <raw_file> --replay-rows <rows_file> --replay-ranges <ranges_file>
python finedge.py configure --startup <startup_directory> --firmware-release <release_json>
python finedge.py native
python finedge.py demo
```

密码通过当前终端环境变量`FINEDGE_BOARD_PASSWORD`提供。串口使用`configure --serial-port`指定的设备和`--baud`波特率。第一次启动会读取Linux boot_id和SD固件哈希，建立本机状态。随后正常软件reboot，核验boot_id变化及干净控制器，再创建新SDK所有者。完整输入预检发生在访问串口之前。

主机将模型、原生C代码和独立Python3.8板端程序组合成运行包。板端程序通过本地相邻模块导入，独立于主机仓库包结构。界面启动后按所选历史事件时钟派发MBO，保存源时钟、预测、计数和分析结果。

## 回放与状态

`python finedge.py demo --review <run_directory>`打开已有板测记录。界面包含盘口K线、预测、实际方向兑现、5ms覆盖率、H50参考及时率和窗口身份。空闲界面、记录复盘及真实板卡运行分别显示状态。

`.local/board/`保存板卡锁、请求队列和加载固件记录，`.local/runs/`保存每次运行，`.local/dashboard/artifacts/`保存界面URL和最近记录位置。恢复过程优先采用正常软件重启。板卡失去管理响应时保留现场，由实际故障决定恢复方式。

## 数据与训练入口

```shell
python finedge.py data --help
python finedge.py train --help
```

`data`提供已有ITCH审计、重建与检查命令，配置位于`config/research/`。`train`接入H50注册式训练入口，需要明确的注册文件、哈希、样本缓存和预算记录。公开仓库提供人工合成数学/结构测试及冻结权重。实际训练数据与登记收据按数据来源独立准备。
