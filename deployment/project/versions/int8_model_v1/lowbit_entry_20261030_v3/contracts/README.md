# 契约目录说明

**唯一外部入口规范是上一级 `input_contract.json`**，其内容在 `frozen_candidate.json` 冻结。`published.json`给出当前发布与质量状态。公共r1未被D修改。

`study_plan.json`是最初的预声明；`plan_amendment.json`记录编译失败及SDK边界验证之后、生成新评价集之前做出的改变。最终采用固定每特征对称INT8、NCHWc16、SDK实际舍入，未采用最初affine/time-major方案。

所有 `quantization_int8*.json` 是**历史编译尝试/比例载体**，其中shape、prefix、RNE等字段可能描述失败方案，不能当最终接口使用。`codec3.py`仅从 `quantization_int8_folded.json` 取40个scale；这40个数必须与冻结 `input_contract.json` 数值及 `artifacts/int8_folded/scales.f32`逐位一致，frozen_check和发布probe会强制验证。

冻结文件中的“尚未生成新评价”描述冻结发生时的状态，不是当前进度。当前进度见上级 `quality_report.json`：新评价已完成且预算通过，真实板测仍未完成。
