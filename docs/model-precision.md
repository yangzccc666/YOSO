# 训练与部署精度

数据集划分仅负责样本分配与标签转换，不设置 INT8/FP16/FP32，也不会改变图片质量。

## 在哪里设置

| 环节 | 设置 | 实际作用 |
| --- | --- | --- |
| YOLO 模型训练 | 训练计算精度：AMP / FP32 | `amp=True` 或 `amp=False` 传入 YOLO；不是 INT8 训练 |
| Jetson ONNX 导出 | 后续部署精度：INT8 / FP16 / FP32 | ONNX 始终按 FP32 导出；选择高精度时不再创建 INT8 量化待办 |
| Docker ONNX 量化 | 量化操作 | INT8 部署路线的量化步骤；高精度路线跳过 |
| 远程 TensorRT 引擎构建 | 构建精度：FP16 / FP32 / 自动最佳 | 未量化 ONNX 可以构建高精度引擎；FP32 禁用 TF32；自动最佳偏向性能 |
| 本地 STAR 打包 | 引擎精度声明 | 仅声明 PLAN 的精度，不转换或恢复权重 |

现有默认值和已保存的配置不被强制重置：训练继续默认 AMP；导出继续默认走 INT8 量化待办；构建和打包保留已有选择。选择新的精度后可保存为默认值。

## 推荐的对照流程

1. 先用现有 PT，不必为了精度对照重新划分或重新训练。
2. Jetson ONNX 导出选择后续部署精度 FP16 或 FP32，生成未量化 ONNX，跳过 Docker 量化。
3. 将 ONNX 手动放到盒子上，在构建功能中输入盒子上的路径，选择对应 FP16 或 FP32。平台不把本机 ONNX 路径误当作远端路径自动填入。
4. 使用不同源文件名或目录保留 INT8/FP16/FP32 对照结果，避免开启覆盖后丢失旧引擎。
5. STAR 打包选择与实际引擎对应的精度声明；在同一批代表性图片/视频上，固定输入尺寸、阈值和前后处理，比较漏检、误检、框的位置和速度。

优先测试 FP16 的性能/数值精度折中，再用 FP32 做对照；若 PT 本身已经漏检，单纯提高部署精度通常不能替代样本、标注和训练优化。FP32 不保证比 AMP 训练的模型检出率更高。需要训练数值对照时可关闭 AMP，并按显存降低 batch。

## 检查与限制

FP16/FP32 构建前用盒子上的 python3 检查 ONNX 结构，检查脚本只依赖标准库，跳过权重原始字节以节省内存。检测 Q/DQ、QLinear、Integer 等量化算子时拒绝高精度构建；FP32 还拒绝已含 FP16/BF16/低精度浮点声明的输入。检查失败不会静默放行。自定义插件的内部精度和最终 TensorRT 每层执行精度仍需通过实际引擎检查，不能靠文件名或 STAR 元数据证明。

已经量化的 ONNX/PLAN/STAR 丢失的信息不能靠改下拉菜单恢复。必须从原始 PT 重新导出未量化模型。当前引擎构建参数针对现有 Jetson 的 TensorRT 8/10 工作流；不承诺 TensorRT 11 强类型构建流程兼容。

参考：[Ultralytics 训练参数](https://docs.ultralytics.com/modes/train/)、[NVIDIA TensorRT 量化类型](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)、[trtexec 参数](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)。
