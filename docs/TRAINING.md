# H50训练复现

## 入口和输入

`finedge.py train`提供独立的`prepare`、`admit`和`run`入口。它们从显式文件和SHA256读取输入，输出写入选择的新目录。路径可使用资源URI、绝对路径，或相对元数据文件的路径。

训练使用5层因果TCN、40通道、100条滑窗和H50标签。模型启用weight normalization，共33539个可训练参数。优化器为AdamW，学习率0.001、weight decay为0.0001、batch为256、seed为42。每轮使用`seed + epoch`的PCG64排列。验证按准确率优先、Macro-F1次优的精确分数选取模型，平分保留较早轮次，连续3轮没有改善则停止，最多10轮。类别依次为跌、平、涨，决策使用原始argmax。

原始订单适配、筛选、H50标签和数据索引源码位于`software/data/`与`software/training/tcn_research_data_p2.py`。本入口使用已经完成索引及标签的P2数据清单，其格式由`validate_prepared`校验。清单包含以下字段。

| 字段 | 内容 |
|---|---|
| version、status | TCN-research-B-P2-data-1.0、prepared_real_data_passed |
| horizon、threshold | 50、0 |
| sources | 12月8—12日TRAIN、15—16日开发验证的7个352字节记录文件及哈希 |
| ranges | 连续段边界、角色、100条输入和50条未来范围、窗口数、标签偏移 |
| day_label_reports | 逐日u8标签文件、字节数、哈希和三类计数 |
| normalization | TRAIN原始40维统计，用于核对数量通道的一致性 |
| sample_counts、train_class_counts、tuning_class_counts | 每个角色的窗口总数和标签计数 |
| 来源与筛选状态 | 源记录一致性、日期范围及数据覆盖记录 |

`RECORD_FIELDS`定义352字节记录格式，包含序号、时间戳、连续段、状态、档位数及`uint64[40]`价量。每段N条记录产生`max(0,N-149)`个带H50标签的训练窗。最后50条为标签未来范围。

训练数据和清单由数据持有方准备。原模型训练使用的12月8—12日完整行情、原始登记收据及完整缓存需要向训练交付方取得。随仓库测试使用7个人工小样本文件，并明确记录`data_kind=synthetic`。

## 准备相对价缓存

```shell
python -B finedge.py train prepare --data-manifest <P2_manifest.json> --data-sha256 <SHA256> --output <new_cache_directory>
```

入口复用交付的相对价预处理。每条记录以自身最优买卖价计算中间价，价格变为相对中间价，数量保持原值。统计仅拟合TRAIN日期，使用float64计算均值和总体标准差，归一化后保存float32逐条缓存。数量通道统计与P2统计核对。原始记录和标签文件在准备前后核对哈希。

输出包含逐日缓存、归一化参数、独立统计核验和`cache_manifest.json`。终端返回清单SHA256。缓存只保存逐条40维数值，训练时按段内端点组窗。

## 准入和训练

```shell
python -B finedge.py train admit --data-manifest <P2_manifest.json> --data-sha256 <SHA256> --cache-manifest <cache_manifest.json> --cache-sha256 <SHA256>
python -B finedge.py train run --data-manifest <P2_manifest.json> --data-sha256 <SHA256> --cache-manifest <cache_manifest.json> --cache-sha256 <SHA256> --output <new_run_directory> --device cpu --epochs 10
```

`admit`检查日期、连续段、窗口和标签计数、归一化、缓存与标签文件哈希，以及缓存对P2清单的来源绑定。缓存文件放在缓存清单目录内，标签放在数据清单目录内，支持相应子目录。

`run`通过准入后从随机初始化开始训练。`--device cuda`要求当前Python环境具备相应CUDA版PyTorch。设备、实际库版本和训练设置写入结果。CPU与CUDA运行分别记录身份。

每轮保存`epochNN.pt`，当前最佳权重保存为`best.pt`，结果保存为`result.json`，归一化另存为`normalization.json`。检查点包含模型、优化器、选取历史、归一化和输入哈希，结果另记源码哈希、初始化身份、优化器步数及成功或失败状态。已有输出目录会保留，并要求选择新目录。

原始注册式训练代码继续位于`tcn_existing_data_ed4_train.py`，用于查看历史训练的预算、登记和验收机制。`--registration`参数保留原接口，并要求其原始协议和登记收据。当前公开目录的可执行训练路线使用本页三条入口。

## 真实执行的自测

自动测试通过统一CLI完成缓存准备、准入、一个CPU训练轮次和检查点重载。7个人工连续段共30个训练窗和12个开发验证窗，实际执行1次优化器更新，重载33539参数模型后状态哈希一致，并核验权重相对初始化发生变化。还覆盖元数据或标签修改、父清单不匹配、相对路径迁移和已有目录保护。

```shell
python -B -m unittest tests.test_training_entry -v
```

完整训练日的重新训练需要配套授权行情和清单。本轮执行的是上述合成小样本训练链路。已发布冻结模型的数值与板测验收分别见`MODEL_BUILD.md`和`BOARD_VALIDATION.md`。
