#!/bin/bash
# RAG评估系统启动脚本

echo "🚀 启动RAG评估系统..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

# 检查必要的包
echo "📦 检查Python包..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "❌ Streamlit未安装，正在安装..."
    pip3 install streamlit
}

python3 -c "import pandas" 2>/dev/null || {
    echo "❌ Pandas未安装，正在安装..."
    pip3 install pandas
}

# 检查环境变量
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "⚠️  DASHSCOPE_API_KEY 环境变量未设置"
    echo "请设置: export DASHSCOPE_API_KEY=your_api_key"
fi

# 启动Streamlit应用
echo "🌐 启动Web界面..."
echo "访问地址: http://localhost:8501"
echo "按 Ctrl+C 停止服务"

streamlit run app.py --server.port 8501 --server.address 0.0.0.0