# 模型构建与验证

## 冻结规范

H50模型为ED4 feature42 seed42的epoch3，输入`float32[1,40,100]`，输出`[1,3]`。归一化参数位于`models/normalization.json`。CPU检查点的路径元数据已转换为资源引用，逐张量核对保留全部138个张量。

公开NPU模型使用原生INT8布局`[1,3,1,100,16]`，4800字节。40通道扩展为48通道，补位800字节为零。各通道尺度冻结在`models/npu/scales.f32`，正式原生编译入口复用这些尺度。

## 可执行阶段

| 命令 | 输入 | 输出 |
|---|---|---|
| model export | 冻结TorchScript及32窗合成参考 | .local/runtime/export/ONNX与验证报告 |
| model polyphase --label NAME --preserve-taps | 上一步ONNX | .local/runtime/subhalf/NAME/等价图与96窗验证 |
| model calibration --input FILE | 明确提供的归一化float32[N,40,100] | 编译校准二进制、列表及来源哈希 |
| model compile --kind tf32 | 随仓库冻结的polyphase图 | Icraft解析、优化、TF32转换、适配、指令生成 |
| model compile --kind int8 | 同图与明确校准输入 | Icraft INT8编译结果与阶段日志 |
| model compile --kind mixed | INT8结果及逐层诊断收据 | 首尾敏感层保留高精度的混合图 |
| model compile --kind native | 混合图、原生C++工具及冻结尺度 | 首层尺度折叠、原生INT8输入布局与指令 |

各编译阶段结果保存在`.local/model_build/{base,mixed,native}/`，C++源码留在`deployment/icraft/`相应目录。CMake通过集中配置选择，SDK从`--icraft`指定位置加载。完整混合精度链需要逐层诊断和冻结收据，入口先检查必要文件并给出缺项位置。

## 中间阶段与历史收据

混合精度选层逻辑位于`deployment/icraft/mixed/diagnose.py`与`compile_mixed.py`。原生尺度折叠、张量布局和Host核验位于`deployment/icraft/native/`。这些阶段保留模型身份、输入尺度、输出层数及失败记录的核对。

迁移验收已运行ONNX导出和图变换。完整Icraft重编译及实际NPU数值验收保留为待完成项，当前部署工件由`models/identity.json`绑定。

## 定点ROM

```shell
python -B -m deployment.quantization.generate_fixed_rom --out .local/rom_new
```

ROM生成器使用冻结归一化参数及INT8尺度，计算整数比较阈值和舍入边界。输出四份阈值表及位宽参数。迁移验收将新输出逐字节对照`hardware/rom/`中五份文件。
