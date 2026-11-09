# 模型部署

`export/`提供ONNX导出和因果图变换，`icraft/`保存base、mixed、native阶段及C++工具，`quantization/`生成整数ROM。

统一入口为`python finedge.py model --help`。阶段输入、输出和未完成验收见`docs/MODEL_BUILD.md`。
