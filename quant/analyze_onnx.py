import onnx
from collections import Counter
import argparse


def analyze_onnx_nodes(onnx_path, target_op="Identity"):
    print(f"正在解析模型: {onnx_path} ...\n")
    
    # 1. 加载 ONNX 模型
    # 注意：如果模型非常大（>2GB），需要使用 load_external_data=False 相关的特殊加载方式，常规模型直接 load 即可
    model = onnx.load(onnx_path)
    print(f"总共算子数量：{len(model.graph.node)}")
    
    # 2. 遍历计算图中的所有节点，统计各 op_type 的数量
    op_counts = Counter()
    
    for node in model.graph.node:
        op_counts[node.op_type] += 1
    
        
    # 3. 输出目标算子的统计结果
    target_count = op_counts.get(target_op, 0)
    print(f"🎯 统计结果: 模型中共找到 【 {target_count} 】 个 '{target_op}' 算子。")
    
    # 4. (附加功能) 打印按数量排序的所有算子分布
    print("-" * 30)
    print("模型整体算子类型分布 (按数量排序):")
    for op, count in op_counts.most_common():
        # 把目标算子高亮显示一下，方便肉眼看
        if op == target_op:
            print(f" -> {op}: {count}  <--- (你的目标)")
        else:
            print(f"    {op}: {count}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='', help='onnx model path')
    parser.add_argument('--target_op', type=str, default='Identity', help='target operator to analyze')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    analyze_onnx_nodes(args.onnx_path, args.target_op)
