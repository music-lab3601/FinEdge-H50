# H50软件源码

训练实现位于`project/scripts/`，保留原模块的相对层级。主入口为`tcn_existing_data_ed4_train.py`，特征为`tcn_existing_data_ed4_features.py`，模型定义为`tcn_candidate.py`和`tcn_research_p5_model.py`。

最近100条十档盘口预测未来第50条记录的端点中间价方向。价格采用逐记录相对中间价，数量保留原值，归一化使用冻结TRAIN统计。

当前可直接核验的冻结模型与人工合成样本位于根目录`models/cpu/`。模型及源码公开副本的身份见`models/identity.json`和`provenance/source_map.json`。

本目录其余交付说明和部分阶段记录保留来源上下文，具体公开范围以根目录说明为准。训练入口的数据配置及目录迁移在后续复现阶段执行。
