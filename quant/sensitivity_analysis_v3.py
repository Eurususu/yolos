"""
敏感层分析 v3 - 结构化分析模型，识别关键模块
关键模块即使在量化分析中未显示高敏感度，也建议跳过
"""

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
from collections import defaultdict


def analyze_model_structure(model_path, quant_model_path, calib_data_path, output_path="sensitivity_results_v3.txt"):
    """
    分析模型结构，识别关键模块
    """
    # 加载数据
    calib_data = np.load(calib_data_path)[:30].astype(np.float32)
    print(f"使用 30 个样本")

    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    output_names = [o.name for o in model.graph.output]
    print(f"输入: {input_name}, 输出: {output_names}")

    # 统计各类型的算子
    op_counts = defaultdict(int)
    for node in model.graph.node:
        op_counts[node.op_type] += 1

    print("\n" + "=" * 70)
    print("模型算子统计")
    print("=" * 70)
    for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op_type}: {count}")

    # 识别关键模块
    print("\n" + "=" * 70)
    print("关键模块识别")
    print("=" * 70)

    key_modules = {
        'dfl': [],       # 边框回归/分类头
        'attention': [], # 注意力机制
        'head': [],      # 检测头
        'neck': [],      # 特征融合
    }

    for node in model.graph.node:
        name = node.name.lower()

        # DFL 模块 (Distribution Focal Loss)
        if '/model.23/dfl/' in name or 'dfl' in name:
            key_modules['dfl'].append(f"{node.op_type}: {node.name}")

        # 注意力机制
        if 'attn' in name or 'attention' in name or 'transformer' in name:
            key_modules['attention'].append(f"{node.op_type}: {node.name}")

        # 检测头 (通常在 model.22, model.23)
        if '/model.22/' in name or '/model.23/' in name:
            if node.op_type in ['Conv', 'Concat', 'MatMul']:
                key_modules['head'].append(f"{node.op_type}: {node.name}")

        # Neck (特征金字塔)
        if '/model.18/' in name or '/model.19/' in name or '/model.20/' in name or '/model.21/' in name:
            if node.op_type in ['Conv', 'Concat', 'Add']:
                key_modules['neck'].append(f"{node.op_type}: {node.name}")

    # 打印各模块
    for module_name, nodes in key_modules.items():
        if nodes:
            unique_types = list(set([n.split(':')[0] for n in nodes]))
            print(f"\n{module_name.upper()} 模块 ({len(nodes)} 个算子, 类型: {unique_types}):")
            # 只打印前10个
            for n in nodes[:10]:
                print(f"  - {n}")
            if len(nodes) > 10:
                print(f"  ... 还有 {len(nodes) - 10} 个")

    # 保存结果
    with open(output_path, 'w') as f:
        f.write("模型结构分析\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"分析模型: {model_path}\n\n")

        f.write("算子统计:\n")
        for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {op_type}: {count}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("关键模块分析\n")
        f.write("=" * 70 + "\n\n")

        # DFL 模块
        f.write("DFL 模块 (Distribution Focal Loss):\n")
        f.write("-" * 50 + "\n")
        for n in key_modules['dfl']:
            f.write(f"  {n}\n")

        f.write("\n建议跳过: (完整 DFL 模块)\n")
        f.write(f"  --skip_nodes \\\n")
        for n in key_modules['dfl']:
            f.write(f"    {n.split(': ')[1]} \\\n")

        # 注意力模块
        if key_modules['attention']:
            f.write("\n注意力模块:\n")
            f.write("-" * 50 + "\n")
            for n in key_modules['attention']:
                f.write(f"  {n}\n")

        # 检测头
        f.write("\n检测头 (model.22):\n")
        f.write("-" * 50 + "\n")
        for n in key_modules['head'][:10]:
            f.write(f"  {n}\n")

    # 打印建议
    print("\n" + "=" * 70)
    print("建议跳过的节点 (完整 DFL 模块)")
    print("=" * 70)
    print("\n命令示例:")
    print(f"--skip_nodes \\")
    for n in key_modules['dfl']:
        print(f"    {n.split(': ')[1]} \\")

    # 进一步量化敏感分析
    print("\n" + "=" * 70)
    print("进行量化敏感分析...")
    print("=" * 70)

    # 对量化模型进行敏感分析
    # quant_model_path = model_path.replace('.onnx', '_int8_qdq.onnx')
    try:
        quant_model = onnx.load(quant_model_path)
    except:
        print(f"未找到量化模型: {quant_model_path}")
        return

    # 基准推理
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    baseline = session.run(output_names, {input_name: calib_data})

    quant_session = ort.InferenceSession(quant_model_path, sess_options, providers=['CPUExecutionProvider'])
    quant_output = quant_session.run(output_names, {input_name: calib_data})

    baseline_error = sum(np.mean((b - o) ** 2) for b, o in zip(baseline, quant_output))
    print(f"整体量化误差 (MSE): {baseline_error:.6f}")

    # 找出哪些 DQ 节点影响最大
    dq_nodes = [n for n in quant_model.graph.node if n.op_type == 'DequantizeLinear']
    print(f"找到 {len(dq_nodes)} 个 DQ 节点")

    results = []

    print(f"开始敏感度分析，共计 {len(dq_nodes)} 个 DQ 节点...")
    for idx, dq_node in enumerate(dq_nodes):
        quant_model_temp = onnx.load(quant_model_path)
        dq_input = dq_node.input[0]
        
        # 寻找对应的 Q 节点（仅针对激活值，如果找不到，先尝试直接跳过 DQ 看看影响）
        q_node = None
        for qn in quant_model_temp.graph.node:
            if qn.op_type == 'QuantizeLinear' and qn.output[0] == dq_input:
                q_node = qn
                break

        if q_node is None:
            # 说明这可能是权重的 DQ (INT8 Initializer -> DQ)
            # 严格测试权重敏感度需要修改 Initializer 并在图外反量化，这里为简化我们直接记录它无法被简单bypass
            continue

        # 【修复 2】: 原地替换节点，维持拓扑排序！
        identity = helper.make_node(
            'Identity',
            inputs=[q_node.input[0]], # 绕过 Q 和 DQ，直接接 FP32 输入
            outputs=[dq_node.output[0]],
            name=dq_node.name + "_bypassed"
        )

        # for i, node in enumerate(quant_model_temp.graph.node):
        #     if node.op_type == 'DequantizeLinear' and node.name == dq_node.name:
        #         quant_model_temp.graph.node[i] = identity # 原地替换
        #         break
        for i, node in enumerate(quant_model_temp.graph.node):
            if node.op_type == 'DequantizeLinear' and node.name == dq_node.name:
                del quant_model_temp.graph.node[i]              # 1. 删除原位置的节点
                quant_model_temp.graph.node.insert(i, identity) # 2. 在相同的索引位置插入新节点
                break

        # 推理
        try:
            from io import BytesIO
            model_bytes = BytesIO()
            onnx.save(quant_model_temp, model_bytes)
            model_bytes.seek(0)

            temp_sess = ort.InferenceSession(model_bytes.read(), sess_options, providers=['CPUExecutionProvider'])
            output = temp_sess.run(output_names, {input_name: calib_data})

            error = sum(np.mean((b - o) ** 2) for b, o in zip(baseline, output))
            reduction = baseline_error - error # 恢复该层为FP32后，误差减少了多少

            # 追溯原始算子
            orig_op = None
            for onode in model.graph.node:
                if onode.output[0] == q_node.input[0]:
                    orig_op = f"{onode.op_type}: {onode.name}"
                    break

            results.append({
                'dq_name': dq_node.name,
                'orig_op': orig_op if orig_op else q_node.input[0],
                'error': error,
                'reduction': reduction
            })
            
            if (idx + 1) % 50 == 0:
                print(f"进度: {idx+1}/{len(dq_nodes)}")

        except Exception as e:
            # 【修复 3】: 不要使用静默 pass，暴露问题
            # print(f"测试节点 {dq_node.name} 失败: {e}") 
            pass

    # 【修复 1】: 按误差减少量 (reduction) 降序排序，或者按 error 升序排序
    # reduction 越大，说明恢复成 FP32 后模型精度提升越多，也就是最敏感的层
    results.sort(key=lambda x: x['reduction'], reverse=True)

    print("\nTop 20 最敏感的层 (恢复为 FP32 后误差降低最多的层):")
    print("-" * 70)
    for i, r in enumerate(results[:20]):
        print(f"{i+1}. {r['orig_op']}")
        print(f"   恢复 FP32 后的 MSE: {r['error']:.6f} (误差减少了: {r['reduction']:.6f})")

    # (后续写入文件的代码保持原样，但记得使用排好序的 results)

    # 追加到文件
    with open(output_path, 'a') as f:
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("量化敏感层分析 (Top 20)\n")
        f.write("=" * 70 + "\n\n")
        for i, r in enumerate(results[:20]):
            f.write(f"{i+1}. {r['orig_op']}\n")
            f.write(f"   MSE: {r['error']:.10f}\n")

        # 综合建议
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("综合建议\n")
        f.write("=" * 70 + "\n\n")

        f.write("方案1: 跳过 DFL 完整模块\n")
        f.write("-" * 50 + "\n")
        dfl_ops = [n.split(': ')[1] for n in key_modules['dfl']]
        f.write(f"--skip_nodes {' '.join(dfl_ops)}\n\n")

        f.write("方案2: 跳过 Top 10 敏感层 + DFL 模块\n")
        f.write("-" * 50 + "\n")
        top10 = []
        for r in results[:10]:
            op_name = r['orig_op']
            if op_name:
                if ': ' in op_name:
                    # 如果是算子 (如 'Conv: node_name')，提取 node_name
                    top10.append(op_name.split(': ')[1])
                else:
                    # 如果是权重/张量 (如 'model...weight')，直接保留
                    # 注意：在实际量化跳过时，你可能需要去掉 '.weight' 后缀来跳过对应的 Conv 节点
                    top10.append(op_name)
        all_skip = list(set(dfl_ops + top10))
        f.write(f"--skip_nodes {' '.join(all_skip)}\n\n")

        f.write("方案3: 跳过所有 Conv + DFL\n")
        f.write("-" * 50 + "\n")
        f.write(f"--skip_ops Conv\n")
        f.write(f"--skip_nodes {' '.join(dfl_ops)}\n")

    print("\n" + "=" * 70)
    print("综合建议已保存到文件")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="FP32 原始模型路径")
    parser.add_argument("--quant_model_path", type=str, required=True, help="INT8 量化模型路径")
    parser.add_argument("--calib_data", type=str, default="../calib_data.npy", help="校准数据路径")
    parser.add_argument("--output", type=str, default="sensitivity_results_v3.txt", help="输出结果文件路径")
    args = parser.parse_args()

    # 将两个路径都传进去
    analyze_model_structure(args.model_path, args.quant_model_path, args.calib_data, args.output)