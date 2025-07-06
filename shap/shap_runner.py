# shap/shap_runner.py
import sys, os
import subprocess
import numpy as np

# 确保能导入项目根目录下的模块
sys.path.insert(0, os.path.abspath('..'))
from config import config

# 要依次执行的脚本（相对于当前 shap/ 目录）
scripts = [
    "1_load_model.py",
    "2_prepare_data.py",
    "3_run_shap.py",
    "4_visualize_shap.py"
]

# 运行单个脚本的辅助函数
# interactive=True 时不捕获输出，允许交互式输入
def run(script, interactive=False):
    print(f"\n>>> 正在执行：{script}")
    cmd = [sys.executable, script]
    if interactive:
        # 直接运行，输出实时打印
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            sys.exit(proc.returncode)
    else:
        # 捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 执行失败！错误信息如下：")
            print(result.stderr)
            sys.exit(1)
        else:
            # 打印子进程标准输出
            print(result.stdout.strip())

# 校验生成文件完整性的函数
def validate_outputs():
    required_inputs = ["X_val.npy", "X_tab_sample.npy"]
    # 如果有图像分支
    if os.path.exists("X_img_sample.npy"):
        required_inputs.append("X_img_sample.npy")
    missing = [f for f in required_inputs if not os.path.exists(f)]
    if missing:
        print(f"❌ 缺少必要输入文件：{missing}")
        sys.exit(1)

    shap_outputs = []
    if os.path.exists("shap_tab_values.npy"): shap_outputs.append("shap_tab_values.npy")
    if os.path.exists("shap_img_values.npy"): shap_outputs.append("shap_img_values.npy")
    if not shap_outputs:
        print("❌ 未生成任何 SHAP 输出文件，请检查 3_run_shap.py 的执行情况")
        sys.exit(1)

    print(f"[✓] 输入和 SHAP 输出文件均已生成：\n  inputs: {required_inputs}\n  outputs: {shap_outputs}")


if __name__ == "__main__":
    print("🚀 开始运行 SHAP 分析流程...")

    # 解析命令行参数
    full_visualization = False
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--full", "-f"):
            full_visualization = True
            print("模式: 完整可视化（包含所有图表和交互式HTML）")
        elif sys.argv[1] in ("--help", "-h"):
            print("用法:")
            print("  python shap_runner.py       # 基础模式")
            print("  python shap_runner.py --full # 完整可视化模式")
            sys.exit(0)

    # 执行流程
    for idx, script in enumerate(scripts):
        if script == "3_run_shap.py":
            run(script, interactive=True)
        else:
            run(script)

        if idx == 2:  # 在第三步后验证
            validate_outputs()

    # 在完整模式下添加额外可视化
    if full_visualization:
        print("\n🔍 生成额外可视化...")
        extra_script = "4_visualize_shap.py"  # 确保它包含所有新增的可视化代码
        run(extra_script)

    print("\n[✅] 全部 SHAP 分析步骤已完成")
    print("生成的文件:")
    subprocess.run(["ls", "-lh", "shap_*", "X_*"], check=False)
    if full_visualization and os.path.exists("shap_img_plots"):
        print("\n图像SHAP可视化:")
        subprocess.run(["ls", "-lh", "shap_img_plots"], check=False)
