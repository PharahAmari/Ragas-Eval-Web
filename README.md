# 🔍 RAG评估系统

这是一个集成的RAG（检索增强生成）评估系统，提供两种数据集生成方式和全面的评估功能。

## ✨ 功能特性

### 📁 方法1：自动生成数据集
- 上传多种格式文档（PDF、Word、Excel、TXT等）
- 自动生成问题、答案和上下文
- 使用AI模型创建高质量评估数据集

### 📄 方法2：特定格式文件处理
- 上传包含"问题"列的Excel文件
- 并行获取AI答案和上下文信息
- 转换为标准Ragas评估格式

### 📊 综合评估功能
- 支持6种评估指标：
  - 忠实度 (Faithfulness)
  - 答案相关性 (Answer Relevancy)
  - 上下文精确度 (Context Precision)
  - 上下文召回率 (Context Recall)
  - 答案相似度 (Answer Similarity)
  - 答案正确性 (Answer Correctness)
- 可选择性评估指标
- 详细的结果分析和解读

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd EvalDeploy

# 安装依赖
pip install -r requirements.txt

# 设置API密钥
export DASHSCOPE_API_KEY="your_api_key_here"
```

### 2. 启动应用

#### 方式1：使用启动脚本（推荐）
```bash
./run_app.sh
```

#### 方式2：直接运行
```bash
streamlit run app.py
```

### 3. 访问系统
打开浏览器访问：`http://localhost:8501`

## 📋 使用指南

### 第一步：数据集生成
在"📁 数据集生成"选项卡中选择合适的方法：

**方法1：自动生成数据集**
1. 选择"方法1: 自动生成数据集"
2. 上传文档文件（支持PDF、Word、Excel、TXT等）
3. 设置每个文档生成的问题数量
4. 点击"开始生成数据集"
5. 系统将自动：
   - 解析文档内容
   - 生成相关问题
   - 创建对应答案
   - 提取相关上下文

**方法2：处理特定格式文件**
1. 选择"方法2: 处理特定格式文件"
2. 准备Excel文件，包含"问题"列
3. 上传Excel文件
4. 点击"开始处理文件"
5. 系统将自动：
   - 并行获取AI回答
   - 提取相关上下文
   - 转换为Ragas格式

### 第二步：评估配置
1. 在"⚙️ 评估配置"选项卡中查看指标说明
2. 选择需要的评估指标：
   - 基础指标：推荐全选
   - 高级指标：根据需要选择
3. 点击"开始评估"

### 第三步：结果查看
1. 在"📊 结果查看"选项卡中查看详细结果
2. 支持导出JSON格式结果
3. 提供结果解读和建议

## 📁 项目结构

```
EvalDeploy/
├── app.py                          # 主应用文件
├── multi_file_datasets_generator.py # 多文件数据集生成器
├── get_answer_parallel.py          # 并行获取AI答案
├── get_contexts_parallel.py        # 并行获取上下文
├── convert_to_ragas_formats.py     # Ragas格式转换
├── evaluate_dataset.py             # 数据集评估
├── requirements.txt                # Python依赖
├── run_app.sh                     # 启动脚本
└── README.md                      # 项目说明
```

## 🔧 配置说明

### 环境变量
- `DASHSCOPE_API_KEY`: 通义千问API密钥（必需）

### API配置
系统使用以下API端点：
- 答案生成：`https://aiagent-server.x.digitalyili.com/oapi/assistant/v1/session/run/4bcb486f-86e5-4502-a40b-fdfd976452ce`
- 上下文检索：`https://aiagent-server.x.digitalyili.com/oapi/assistant/v1/session/run/95663b0c-5655-44c0-a6cd-24b3a1de29c6`

### 支持的文件格式

#### 方法1（自动生成）
- PDF (.pdf)
- Word文档 (.docx)
- Excel表格 (.xlsx, .xls)
- 文本文件 (.txt, .md)
- CSV文件 (.csv)
- JSON文件 (.json)

#### 方法2（特定格式）
- Excel文件 (.xlsx, .xls)
- 必须包含"问题"列

## 📊 评估指标说明

### 忠实度 (Faithfulness)
- **用途**: 检测AI回答中的幻觉内容
- **评分**: 0-1（越高越好）
- **含义**: 答案中的事实是否都能从上下文中找到支撑

### 答案相关性 (Answer Relevancy)
- **用途**: 评估答案与问题的匹配度
- **评分**: 0-1（越高越好）
- **含义**: 答案是否直接回答了问题，没有无关信息

### 上下文精确度 (Context Precision)
- **用途**: 评估检索质量
- **评分**: 0-1（越高越好）
- **含义**: 排名靠前的上下文是否都与问题相关

### 上下文召回率 (Context Recall)
- **用途**: 评估检索完整性
- **评分**: 0-1（越高越好）
- **含义**: 是否检索到了回答问题所需的所有信息

### 答案相似度 (Answer Similarity)
- **用途**: 语义相似度比较
- **评分**: 0-1（越高越好）
- **含义**: 生成答案与标准答案的语义相似程度

### 答案正确性 (Answer Correctness)
- **用途**: 综合正确性评估
- **评分**: 0-1（越高越好）
- **含义**: 结合准确性和完整性的综合评估

## 🛠️ 故障排除

### 常见问题

1. **API密钥未设置**
   ```bash
   export DASHSCOPE_API_KEY="your_key"
   ```

2. **依赖包安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **文件上传失败**
   - 检查文件格式是否支持
   - 确保文件大小不超过限制

4. **评估过程中断**
   - 检查网络连接
   - 验证API密钥有效性
   - 减少评估指标数量

### 日志查看
系统会在处理日志区域显示详细的处理信息，有助于问题诊断。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。