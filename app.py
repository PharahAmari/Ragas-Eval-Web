#!/usr/bin/env python3
"""
RAG评估应用集成版本
整合所有功能到一个Flask网页应用
"""

import os
import json
import pandas as pd
import streamlit as st
import tempfile
import threading
import time
import pickle
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置页面配置
st.set_page_config(
    page_title="RAG评估系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入现有模块的类和函数
try:
    from get_answer_parallel import query_answer
    from get_contexts_parallel import query_contexts
    from convert_to_ragas_formats import extract_contexts_from_contexts_column
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    MODULES_AVAILABLE = False

# 评估指标信息
METRICS_INFO = {
    'faithfulness': {
        'name': '忠实度 (Faithfulness)',
        'description': '衡量生成的答案是否忠实于给定的上下文，避免产生幻觉内容',
        'range': '0-1（越高越好）',
        'details': '评估答案是否从提供的上下文中推断出来'
    },
    'answer_relevancy': {
        'name': '答案相关性 (Answer Relevancy)',
        'description': '评估生成的答案与给定问题的相关程度',
        'range': '0-1（越高越好）',
        'details': '检查答案是否直接回答了所提出的问题，避免冗余或无关信息'
    },
    'context_precision': {
        'name': '上下文精确度 (Context Precision)',
        'description': '评估检索到的上下文中有用信息的比例',
        'range': '0-1（越高越好）',
        'details': '衡量排名靠前的上下文chunk是否都与问题相关'
    },
    'context_recall': {
        'name': '上下文召回率 (Context Recall)',
        'description': '评估检索系统是否找到了回答问题所需的所有相关信息',
        'range': '0-1（越高越好）',
        'details': '检查标准答案中的信息是否都能在检索到的上下文中找到'
    },
    'answer_similarity': {
        'name': '答案相似度 (Answer Similarity)',
        'description': '评估生成答案与标准答案的语义相似度',
        'range': '0-1（越高越好）',
        'details': '使用语义相似度模型比较AI回答和标准答案'
    },
    'answer_correctness': {
        'name': '答案正确性 (Answer Correctness)',
        'description': '综合评估答案的准确性和完整性',
        'range': '0-1（越高越好）',
        'details': '结合事实准确性和语义相似度的综合评估指标'
    }
}

def init_session_state():
    """初始化session状态"""
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = []
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'detailed_results_df' not in st.session_state:
        st.session_state.detailed_results_df = None
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = None
    if 'last_processed_file_hash' not in st.session_state:
        st.session_state.last_processed_file_hash = None

def log_message(message: str):
    """添加日志消息到session状态"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.processing_log.append(f"[{timestamp}] {message}")
    logger.info(message)

def get_file_hash(uploaded_file):
    """计算上传文件的哈希值"""
    file_content = uploaded_file.getvalue()
    return hashlib.md5(file_content).hexdigest()

def get_directory_size(directory):
    """获取目录大小（MB）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)  # 转换为MB
    except:
        return 0

def cleanup_temp_excel_files():
    """清理遗留的临时Excel文件"""
    try:
        import glob
        temp_dir = tempfile.gettempdir()

        # 查找临时Excel文件
        temp_excel_files = glob.glob(os.path.join(temp_dir, "tmp*.xlsx"))
        current_time = time.time()
        cleaned_count = 0

        for file_path in temp_excel_files:
            try:
                # 删除1小时以前的临时Excel文件
                if current_time - os.path.getmtime(file_path) > 3600:
                    os.unlink(file_path)
                    cleaned_count += 1
            except:
                pass

        if cleaned_count > 0:
            log_message(f"清理了 {cleaned_count} 个遗留的临时Excel文件")
    except Exception as e:
        log_message(f"清理临时Excel文件失败: {str(e)}")

def cleanup_old_state_files():
    """智能清理旧状态文件 - 保护策略，只清理真正需要的文件"""
    try:
        state_dir = os.path.join(tempfile.gettempdir(), "rag_eval_states")
        if not os.path.exists(state_dir):
            return

        # 检查总目录大小
        dir_size_mb = get_directory_size(state_dir)

        current_time = time.time()
        cleaned_count = 0
        files_info = []

        # 收集文件信息
        for filename in os.listdir(state_dir):
            if filename.startswith("state_") and filename.endswith(".pkl"):
                file_path = os.path.join(state_dir, filename)
                try:
                    stat = os.stat(file_path)
                    files_info.append({
                        'path': file_path,
                        'filename': filename,
                        'mtime': stat.st_mtime,
                        'size': stat.st_size,
                        'age_days': (current_time - stat.st_mtime) / (24 * 3600)
                    })
                except:
                    continue

        # 排序：按修改时间排序（最旧的在前）
        files_info.sort(key=lambda x: x['mtime'])

        # 保守的清理策略 - 只在确实需要时清理
        for file_info in files_info:
            should_delete = False

            # 1. 删除超过14天的文件（延长到14天）
            if file_info['age_days'] > 14:
                should_delete = True

            # 2. 如果目录超过100MB，删除7天以上的文件（提高阈值）
            elif dir_size_mb > 100 and file_info['age_days'] > 7:
                should_delete = True

            # 3. 如果文件数量超过50个，删除3天以上的文件（更保守）
            elif len(files_info) > 50 and file_info['age_days'] > 3:
                should_delete = True

            if should_delete:
                try:
                    os.unlink(file_info['path'])
                    cleaned_count += 1
                    dir_size_mb -= file_info['size'] / (1024 * 1024)
                    log_message(f"清理断点文件: {file_info['filename']} (存在 {file_info['age_days']:.1f} 天)")
                except:
                    pass

        if cleaned_count > 0:
            log_message(f"自动清理了 {cleaned_count} 个过期断点文件，目录大小: {dir_size_mb:.1f}MB")
    except Exception as e:
        log_message(f"清理断点文件失败: {str(e)}")

def save_processing_state(file_hash, df, answer_progress, context_progress):
    """保存处理状态到本地文件，优化存储空间"""
    try:
        state_dir = os.path.join(tempfile.gettempdir(), "rag_eval_states")
        os.makedirs(state_dir, exist_ok=True)

        # 每次保存前清理旧文件和遗留的临时文件
        cleanup_old_state_files()
        cleanup_temp_excel_files()

        state_file = os.path.join(state_dir, f"state_{file_hash}.pkl")

        # 优化存储：只保存必要的数据，不保存整个DataFrame
        essential_columns = ['问题', 'AI回答', '参考文档', 'Contexts']
        df_minimal = df[essential_columns].to_dict('records')

        state_data = {
            'file_hash': file_hash,
            'df': df_minimal,  # 只保存关键列
            'answer_progress': list(answer_progress),
            'context_progress': list(context_progress),
            'timestamp': datetime.now().isoformat()
        }

        with open(state_file, 'wb') as f:
            pickle.dump(state_data, f)

        # 检查文件大小
        file_size = os.path.getsize(state_file) / 1024  # KB
        log_message(f"进度已保存：答案 {len(answer_progress)}/总计, 上下文 {len(context_progress)}/总计 (文件大小: {file_size:.1f}KB)")
        return True
    except Exception as e:
        log_message(f"保存进度状态失败: {str(e)}")
        return False

def load_processing_state(file_hash):
    """从本地文件加载处理状态"""
    try:
        state_dir = os.path.join(tempfile.gettempdir(), "rag_eval_states")
        state_file = os.path.join(state_dir, f"state_{file_hash}.pkl")

        if not os.path.exists(state_file):
            return None

        with open(state_file, 'rb') as f:
            state_data = pickle.load(f)

        log_message(f"找到之前的进度记录，时间: {state_data['timestamp'][:19]}")
        return state_data
    except Exception as e:
        log_message(f"加载进度状态失败: {str(e)}")
        return None

def align_data_for_evaluation(df):
    """对齐数据，确保只使用同时拥有答案和上下文的数据"""
    aligned_data = []
    skipped_count = 0

    for index, row in df.iterrows():
        # 检查问题是否存在
        if pd.isna(row['问题']) or str(row['问题']).strip() == "":
            skipped_count += 1
            continue

        question = str(row['问题']).strip()
        answer = str(row.get('AI回答', '')).strip()
        contexts_raw = str(row.get('Contexts', '')).strip()

        # 只有当答案和上下文都存在时才包含该数据
        if answer and contexts_raw and answer != 'nan' and contexts_raw != 'nan':
            try:
                contexts = extract_contexts_from_contexts_column(contexts_raw)
                if contexts and any(ctx.strip() for ctx in contexts):
                    aligned_data.append({
                        'question': question,
                        'answer': answer,
                        'contexts': contexts,
                        'ground_truth': answer
                    })
                else:
                    skipped_count += 1
            except Exception as e:
                log_message(f"处理第 {index + 1} 行数据时出错: {str(e)}")
                skipped_count += 1
        else:
            skipped_count += 1

    if skipped_count > 0:
        log_message(f"数据对齐：跳过 {skipped_count} 条不完整的记录，保留 {len(aligned_data)} 条完整记录")

    return aligned_data

def display_metrics_info():
    """显示评估指标信息"""
    with st.expander("📊 评估指标详细说明", expanded=False):
        st.markdown("### 评估指标概览")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 基础指标")
            for key in ['faithfulness', 'answer_relevancy', 'context_precision']:
                metric = METRICS_INFO[key]
                st.markdown(f"**📈 {metric['name']}**")
                st.markdown(f"- {metric['description']}")
                st.markdown(f"- 取值范围: {metric['range']}")
                st.markdown("")

        with col2:
            st.markdown("#### 🚀 高级指标")
            for key in ['context_recall', 'answer_similarity', 'answer_correctness']:
                metric = METRICS_INFO[key]
                st.markdown(f"**📈 {metric['name']}**")
                st.markdown(f"- {metric['description']}")
                st.markdown(f"- 取值范围: {metric['range']}")
                st.markdown("")

        st.markdown("---")
        st.markdown("💡 **使用建议**: 基础指标适合快速评估，高级指标提供更深入的分析")


def process_method2_file(uploaded_file):
    """处理方法2：处理特定格式文件，支持断点重传"""
    log_message("开始处理Excel文件，获取答案和上下文...")

    # 计算文件哈希用于断点重传
    file_hash = get_file_hash(uploaded_file)
    log_message(f"文件哈希: {file_hash[:8]}...")

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    try:
        # 读取Excel文件
        df = pd.read_excel(temp_path)
        log_message(f"读取到 {len(df)} 条记录")

        # 确保必要列存在
        if 'AI回答' not in df.columns:
            df['AI回答'] = ''
        if '参考文档' not in df.columns:
            df['参考文档'] = ''
        if 'Contexts' not in df.columns:
            df['Contexts'] = ''

        # 尝试加载之前的处理状态
        previous_state = load_processing_state(file_hash)
        answer_progress = set()
        context_progress = set()

        if previous_state:
            # 恢复之前的数据
            saved_df = pd.DataFrame(previous_state['df'])
            answer_progress = set(previous_state['answer_progress'])
            context_progress = set(previous_state['context_progress'])

            log_message(f"断点重传：答案进度 {len(answer_progress)}/{len(df)}, 上下文进度 {len(context_progress)}/{len(df)}")

            # 将已获取的数据合并到当前数据框
            for idx, row in saved_df.iterrows():
                if idx < len(df):
                    if pd.notna(row.get('AI回答')) and str(row.get('AI回答')).strip():
                        df.at[idx, 'AI回答'] = row['AI回答']
                        df.at[idx, '参考文档'] = row.get('参考文档', '')
                        answer_progress.add(idx)

                    if pd.notna(row.get('Contexts')) and str(row.get('Contexts')).strip():
                        df.at[idx, 'Contexts'] = row['Contexts']
                        context_progress.add(idx)

        # 保存更新后的文件
        df.to_excel(temp_path, index=False)

        # 并行处理答案和上下文
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 收集需要处理的任务，排除已完成的任务
        answer_tasks = []
        context_tasks = []

        for index, row in df.iterrows():
            if pd.isna(row['问题']) or str(row['问题']).strip() == "":
                continue

            question = str(row['问题']).strip()

            # 检查是否需要获取AI回答（排除已完成的）
            if index not in answer_progress and (pd.isna(row['AI回答']) or str(row['AI回答']).strip() == ""):
                answer_tasks.append((index, question))

            # 检查是否需要获取上下文（排除已完成的）
            if index not in context_progress and (pd.isna(row['Contexts']) or str(row['Contexts']).strip() == ""):
                context_tasks.append((index, question))

        total_tasks = len(answer_tasks) + len(context_tasks)
        completed_tasks = 0

        # 统计成功获取的数量
        successful_answers = 0
        successful_contexts = 0

        if total_tasks == 0:
            status_text.text("所有数据已完整，无需获取新内容")
            progress_bar.progress(1.0)
        else:
            log_message(f"开始并行处理：{len(answer_tasks)} 个AI回答任务，{len(context_tasks)} 个上下文任务")

        # 并行获取AI回答
        if answer_tasks:
            status_text.text(f"正在并行获取 {len(answer_tasks)} 个AI回答...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                # 提交所有AI回答任务
                future_to_index = {
                    executor.submit(query_answer, question, 1): index
                    for index, question in answer_tasks
                }

                # 处理完成的任务
                save_counter = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        if result['success']:
                            df.at[index, 'AI回答'] = result['ai_answer']
                            df.at[index, '参考文档'] = result['reference']
                            answer_progress.add(index)
                            successful_answers += 1
                    except Exception as e:
                        log_message(f"获取问题 {index + 1} 的AI回答失败: {str(e)}")

                    completed_tasks += 1
                    save_counter += 1
                    progress_bar.progress(completed_tasks / total_tasks)

                    # 每处理10个任务保存一次进度
                    if save_counter % 10 == 0:
                        save_processing_state(file_hash, df, answer_progress, context_progress)
                        df.to_excel(temp_path, index=False)

        # 并行获取上下文
        if context_tasks:
            status_text.text(f"正在并行获取 {len(context_tasks)} 个上下文...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                # 提交所有上下文任务
                future_to_index = {
                    executor.submit(query_contexts, question, 1): index
                    for index, question in context_tasks
                }

                # 处理完成的任务
                save_counter = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        contexts, success = future.result()
                        if success:
                            df.at[index, 'Contexts'] = contexts
                            context_progress.add(index)
                            successful_contexts += 1
                    except Exception as e:
                        log_message(f"获取问题 {index + 1} 的上下文失败: {str(e)}")

                    completed_tasks += 1
                    save_counter += 1
                    progress_bar.progress(completed_tasks / total_tasks)

                    # 每处理10个任务保存一次进度
                    if save_counter % 10 == 0:
                        save_processing_state(file_hash, df, answer_progress, context_progress)
                        df.to_excel(temp_path, index=False)

        # 显示获取成功的统计信息
        if answer_tasks or context_tasks:
            log_message(f"数据获取完成！AI回答: {successful_answers}/{len(answer_tasks)} 成功，上下文: {successful_contexts}/{len(context_tasks)} 成功")
            status_text.text(f"✅ 数据获取完成！AI回答: {successful_answers}/{len(answer_tasks)} 成功，上下文: {successful_contexts}/{len(context_tasks)} 成功")

        time.sleep(2)  # 让用户看到完成统计

        # 最后保存完整的进度状态
        save_processing_state(file_hash, df, answer_progress, context_progress)
        df.to_excel(temp_path, index=False)

        # 转换为Ragas格式，使用数据对齐功能
        log_message("转换为Ragas评估格式，筛选完整数据...")
        ragas_data = align_data_for_evaluation(df)

        # 如果没有有效数据，返回None
        if not ragas_data:
            log_message("错误：没有找到完整的评估数据，请检查Excel文件")
            st.error("没有找到完整的评估数据，请确保Excel文件包含问题、AI回答和上下文数据")
            return None

        log_message(f"成功转换 {len(ragas_data)} 个完整的评估样本")
        st.session_state.last_processed_file_hash = file_hash

        return ragas_data

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def evaluate_partial_data(uploaded_file, selected_metrics: List[str]):
    """评估部分完整的数据，支持断点重传的数据对齐"""
    if not selected_metrics:
        st.error("请选择至少一个评估指标")
        return None

    # 计算文件哈希
    file_hash = get_file_hash(uploaded_file)

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    try:
        # 读取Excel文件
        df = pd.read_excel(temp_path)

        # 尝试加载之前的处理状态
        previous_state = load_processing_state(file_hash)
        if previous_state:
            saved_df = pd.DataFrame(previous_state['df'])
            log_message("使用已保存的进度数据进行评估")
            df = saved_df

        # 使用数据对齐功能获取可评估的数据
        evaluation_data = align_data_for_evaluation(df)

        if not evaluation_data:
            st.warning("没有找到可评估的完整数据，请先获取更多答案和上下文")
            return None

        log_message(f"找到 {len(evaluation_data)} 条可评估的完整数据")

        # 调用原有的评估函数
        return evaluate_dataset(evaluation_data, selected_metrics)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def evaluate_dataset(evaluation_data: List[Dict], selected_metrics: List[str]):
    """评估数据集"""
    if not evaluation_data:
        st.error("没有可评估的数据")
        return None

    # 数据完整性验证
    log_message(f"开始评估数据集验证，共 {len(evaluation_data)} 个样本")

    # 检查数据完整性
    valid_samples = []
    invalid_count = 0

    for i, item in enumerate(evaluation_data):
        # 检查必需字段
        if not item.get('question') or not item.get('answer') or not item.get('contexts'):
            log_message(f"样本 {i+1} 数据不完整，已跳过")
            invalid_count += 1
            continue

        # 检查contexts是否为空列表或包含空字符串
        if not item['contexts'] or all(not ctx.strip() for ctx in item['contexts']):
            log_message(f"样本 {i+1} 上下文数据无效，已跳过")
            invalid_count += 1
            continue

        valid_samples.append(item)

    if invalid_count > 0:
        log_message(f"⚠️ 发现 {invalid_count} 个无效样本，将使用 {len(valid_samples)} 个有效样本进行评估")
        st.warning(f"发现 {invalid_count} 个数据不完整的样本，将使用 {len(valid_samples)} 个有效样本进行评估")

    if not valid_samples:
        st.error("没有有效的评估样本，请检查数据完整性")
        return None

    # 使用有效样本进行评估
    evaluation_data = valid_samples
    log_message(f"开始评估 {len(evaluation_data)} 个有效样本，使用指标: {', '.join(selected_metrics)}")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision,
            context_recall, answer_similarity, answer_correctness
        )
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # 配置LLM
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")
        
        # 临时设置OPENAI_API_KEY环境变量，确保ChatOpenAI能找到
        os.environ["OPENAI_API_KEY"] = api_key
            
        llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_key=api_key,  # 使用openai_api_key参数
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 配置embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            cache_folder="/home/zhangdh17/.cache/huggingface/hub/"
        )

        # 指标映射
        metric_map = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'answer_similarity': answer_similarity,
            'answer_correctness': answer_correctness
        }

        # 选择指标
        metrics = [metric_map[metric] for metric in selected_metrics if metric in metric_map]

        if not metrics:
            st.error("请选择至少一个评估指标")
            return None

        # 创建数据集
        dataset = Dataset.from_list(evaluation_data)
        
        # 执行评估
        start_time = time.time()
        with st.spinner(f"正在评估 {len(evaluation_data)} 个样本，使用 {len(selected_metrics)} 个指标..."):
            result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        
        elapsed_time = time.time() - start_time
        log_message(f"评估完成，耗时: {elapsed_time:.1f} 秒")
        return result

    except Exception as e:
        st.error(f"评估过程中出现错误: {str(e)}")
        log_message(f"评估失败: {str(e)}")
        return None

def display_evaluation_results(results, evaluation_data):
    """显示评估结果"""
    if not results:
        return

    # 获取详细的评估数据DataFrame
    if hasattr(results, 'to_pandas'):
        results_df = results.to_pandas()
    else:
        st.error("无法获取详细评估结果")
        return

    # 数据长度一致性检查
    if len(results_df) != len(evaluation_data):
        st.warning(f"⚠️ 数据长度不一致：评估结果 {len(results_df)} 个，原始数据 {len(evaluation_data)} 个")
        log_message(f"数据长度不一致：评估结果 {len(results_df)} 个，原始数据 {len(evaluation_data)} 个")

        # 截取到较短的长度以避免pandas错误
        min_length = min(len(results_df), len(evaluation_data))
        results_df = results_df.head(min_length)
        evaluation_data = evaluation_data[:min_length]

        st.info(f"已自动调整为 {min_length} 个样本进行结果显示")
        log_message(f"已调整数据长度为 {min_length} 个样本")
    else:
        log_message(f"数据长度一致：{len(results_df)} 个样本")

    # 创建列名映射来处理Ragas返回的列名与METRICS_INFO键名的差异
    column_mapping = {
        'semantic_similarity': 'answer_similarity'  # 映射semantic_similarity到answer_similarity
    }

    # 显示汇总统计
    st.markdown("### 📊 汇总统计")

    # 计算并显示平均分数
    numeric_columns = results_df.select_dtypes(include=['number']).columns
    summary_data = []


    # 为所有指标创建列显示
    metric_cols = st.columns(len(numeric_columns))  # 根据实际指标数量创建列数

    col_idx = 0
    for col in numeric_columns:
        # 使用映射后的列名进行查找
        mapped_col = column_mapping.get(col, col)
        if mapped_col in METRICS_INFO:
            avg_score = results_df[col].mean()
            summary_data.append({
                '指标': METRICS_INFO[mapped_col]['name'],
                '平均分': f"{avg_score:.4f}",
                '百分比': f"{avg_score*100:.2f}%"
            })

            # 在对应的列中显示指标
            if col_idx < len(metric_cols):
                metric_name = METRICS_INFO[mapped_col]['name'].split(' (')[0]  # 提取中文名称
                metric_cols[col_idx].metric(metric_name, f"{avg_score:.4f}")
                col_idx += 1


    # 显示详细结果（每个样本的得分）
    st.markdown("### 📝 详细评估结果")
    
    # 创建完整的结果表格，包含问题、答案和评分
    detailed_data = {
        'question': [item['question'] for item in evaluation_data],
        'answer': [item['answer'] for item in evaluation_data],
        'ground_truth': [item.get('ground_truth', item['answer']) for item in evaluation_data]
    }
    
    # 添加评估分数
    for col in numeric_columns:
        # 使用映射后的列名进行查找
        mapped_col = column_mapping.get(col, col)
        if mapped_col in METRICS_INFO:
            detailed_data[METRICS_INFO[mapped_col]['name']] = results_df[col].tolist()
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # 显示前10行样本结果
    st.markdown("**最多展现前10个样本的详细结果：**")
    st.dataframe(detailed_df.head(10), use_container_width=True)
    
    # 保存完整结果到session state，供下载使用
    st.session_state.detailed_results_df = detailed_df

    # 显示结果解读
    st.markdown("### 📋 结果解读")
    for col in numeric_columns:
        # 使用映射后的列名进行查找
        mapped_col = column_mapping.get(col, col)
        if mapped_col in METRICS_INFO:
            info = METRICS_INFO[mapped_col]
            avg_score = results_df[col].mean()

            # 评估等级
            if avg_score >= 0.8:
                level = "优秀 ✅"
            elif avg_score >= 0.6:
                level = "良好 ⚠️"
            else:
                level = "需要改进 ❌"

            st.markdown(f"**{info['name']}**: {avg_score:.4f} - {level}")
            st.markdown(f"{info['description']}")

def main():
    """主函数"""
    init_session_state()

    st.title("🔍 RAG评估系统")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("🛠️ 系统状态")

        # API配置检查
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            st.success("✅ API密钥已配置")
        else:
            st.error("❌ 请配置DASHSCOPE_API_KEY环境变量")
            st.code("export DASHSCOPE_API_KEY=your_key")

        # 模块状态检查
        if MODULES_AVAILABLE:
            st.success("✅ 核心模块已加载")
        else:
            st.error("❌ 部分模块加载失败")
            st.info("某些功能可能受限")

        # 数据集状态
        if st.session_state.current_dataset:
            dataset_size = len(st.session_state.current_dataset)
            st.info(f"📊 数据集: {dataset_size} 样本")

            # 显示数据集质量检查
            valid_samples = sum(1 for item in st.session_state.current_dataset
                              if item.get('question') and item.get('answer') and item.get('contexts'))
            if valid_samples == dataset_size:
                st.success(f"✅ 数据完整 ({valid_samples}/{dataset_size})")
            else:
                st.warning(f"⚠️ 数据不完整 ({valid_samples}/{dataset_size})")

            # 显示断点重传状态和存储监控
            if st.session_state.last_processed_file_hash:
                state_dir = os.path.join(tempfile.gettempdir(), "rag_eval_states")
                state_file = os.path.join(state_dir, f"state_{st.session_state.last_processed_file_hash}.pkl")
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'rb') as f:
                            state_data = pickle.load(f)
                        answer_progress = len(state_data['answer_progress'])
                        context_progress = len(state_data['context_progress'])
                        st.info(f"🔄 断点状态：答案 {answer_progress}, 上下文 {context_progress}")
                    except:
                        pass

            # 存储空间监控（只读显示）
            state_dir = os.path.join(tempfile.gettempdir(), "rag_eval_states")
            if os.path.exists(state_dir):
                try:
                    dir_size_mb = get_directory_size(state_dir)
                    file_count = len([f for f in os.listdir(state_dir) if f.startswith("state_")])

                    if dir_size_mb > 20:  # 超过20MB时警告
                        st.warning(f"📦 断点存储：{dir_size_mb:.1f}MB ({file_count} 文件) - 系统会自动清理旧文件")
                    elif file_count > 0:
                        st.info(f"📦 断点存储：{dir_size_mb:.1f}MB ({file_count} 文件)")
                except:
                    pass
        else:
            st.warning("📊 暂无数据集")

        # 评估结果状态
        if st.session_state.evaluation_results:
            st.info("📈 评估已完成")
            
            # 显示评估概要
            if hasattr(st.session_state, 'detailed_results_df') and st.session_state.detailed_results_df is not None:
                df = st.session_state.detailed_results_df
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    avg_score = df[numeric_cols].mean().mean()
                    st.metric("平均评分", f"{avg_score:.3f}")
                    
        else:
            st.warning("📈 暂无评估结果")

        # 显示最近的日志
        if st.session_state.processing_log:
            with st.expander("📝 最近日志", expanded=False):
                for log in st.session_state.processing_log[-3:]:  # 显示最近3条
                    st.text(log)

    # 主界面选项卡
    tab1, tab2, tab3 = st.tabs(["📁 数据集生成", "⚙️ 评估配置", "📊 结果查看"])

    with tab1:
        st.header("📁 数据集生成")
        st.markdown("上传包含问题列的Excel文件，系统将获取AI回答和上下文信息")

        st.markdown("---")

        # 文件格式说明
        with st.expander("📋 Excel文件格式要求及新功能"):
            st.markdown("""
            **Excel文件格式**：
            - **问题**: 必须列，包含要评估的问题
            - **标准答案**: 必须列，包含问题的标准答案
            - **AI回答**: 可选列，如果为空将自动获取
            - **参考文档**: 可选列，如果为空将自动获取
            - **Contexts**: 可选列，如果为空将自动获取

            **示例Excel格式**:
            | 问题 | AI回答 | 参考文档 | Contexts |
            |------|--------|----------|----------|
            | 什么是RAG？ | 空或已有答案 | 空或已有 | 空或已有 |
            """)

        # 文件上传
        uploaded_file = st.file_uploader(
            "选择Excel文件",
            type=['xlsx'],
            key="method2_file",
            help="上传包含问题列的Excel文件"
        )

        # 处理选项和断点重传状态显示
        if uploaded_file:
            file_hash = get_file_hash(uploaded_file)
            previous_state = load_processing_state(file_hash)

            col1, col2 = st.columns(2)
            with col1:
                if previous_state:
                    answer_progress = len(previous_state['answer_progress'])
                    context_progress = len(previous_state['context_progress'])
                    st.info(f"🔄 找到断点记录：答案 {answer_progress}，上下文 {context_progress}")
                else:
                    st.info("系统将自动检测并填充缺失的列")
            with col2:
                st.warning("处理可能需要较长时间，请耐心等待")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🚀 开始处理文件", key="process_file", type="primary"):
                if uploaded_file:
                    evaluation_data = process_method2_file(uploaded_file)
                    if evaluation_data:
                        st.session_state.current_dataset = evaluation_data
                        st.success(f"✅ 成功处理 {len(evaluation_data)} 个评估样本")

                        # 显示数据预览
                        st.subheader("📋 数据预览")
                        preview_df = pd.DataFrame(evaluation_data[:5])  # 显示前5个
                        st.dataframe(preview_df, use_container_width=True)
                else:
                    st.warning("请先上传Excel文件")

        with col2:
            if st.button("📊 评估部分数据", key="evaluate_partial", help="评估当前已完整的数据"):
                if uploaded_file:
                    # 简单的指标选择（使用默认指标）
                    default_metrics = ['faithfulness', 'answer_relevancy', 'context_precision']

                    with st.expander("📊 快速评估设置", expanded=True):
                        selected_metrics = st.multiselect(
                            "选择评估指标：",
                            options=['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_similarity'],
                            default=default_metrics,
                            key="partial_eval_metrics"
                        )

                        if st.button("确认评估", key="confirm_partial_eval"):
                            if selected_metrics:
                                results = evaluate_partial_data(uploaded_file, selected_metrics)
                                if results:
                                    st.session_state.evaluation_results = results
                                    # 创建对应的数据集用于结果显示
                                    file_hash = get_file_hash(uploaded_file)

                                    # 创建临时文件读取数据
                                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                                        tmp_file.write(uploaded_file.getbuffer())
                                        temp_path = tmp_file.name

                                    try:
                                        df = pd.read_excel(temp_path)
                                        previous_state = load_processing_state(file_hash)
                                        if previous_state:
                                            df = pd.DataFrame(previous_state['df'])

                                        st.session_state.current_dataset = align_data_for_evaluation(df)
                                        st.success("✅ 部分数据评估完成！")
                                        st.info("👉 请查看结果页面")
                                    finally:
                                        if os.path.exists(temp_path):
                                            os.unlink(temp_path)
                            else:
                                st.warning("请选择至少一个评估指标")
                else:
                    st.warning("请先上传Excel文件")

        with col3:
            if st.button("🔄 清除当前进度", key="clear_progress", help="仅清除当前文件的断点记录，重新开始处理"):
                if uploaded_file:
                    file_hash = get_file_hash(uploaded_file)
                    state_dir = os.path.join(tempfile.gettempdir(), "rag_eval_states")
                    state_file = os.path.join(state_dir, f"state_{file_hash}.pkl")

                    if os.path.exists(state_file):
                        try:
                            os.unlink(state_file)
                            st.success("✅ 已清除当前文件的断点记录")
                            st.info("💡 下次上传此文件将重新开始处理")
                        except Exception as e:
                            st.error(f"清除失败: {str(e)}")
                    else:
                        st.info("当前文件没有断点记录")
                else:
                    st.warning("请先上传Excel文件")


    with tab2:
        st.header("⚙️ 评估配置")

        if st.session_state.current_dataset:
            st.success(f"✅ 已加载数据集，包含 {len(st.session_state.current_dataset)} 个样本")

            # 显示指标说明
            display_metrics_info()

            # 指标选择
            st.subheader("🎯 选择评估指标")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**基础指标**")
                faithfulness_check = st.checkbox("忠实度 (Faithfulness)", key="faithfulness_cb")
                answer_relevancy_check = st.checkbox("答案相关性 (Answer Relevancy)", key="answer_relevancy_cb")
                context_precision_check = st.checkbox("上下文精确度 (Context Precision)", key="context_precision_cb")

            with col2:
                st.markdown("**高级指标**")
                context_recall_check = st.checkbox("上下文召回率 (Context Recall)", value=False, key="context_recall_cb")
                answer_similarity_check = st.checkbox("答案相似度 (Answer Similarity)", value=False, key="answer_similarity_cb")
                answer_correctness_check = st.checkbox("答案正确性 (Answer Correctness)", value=False, key="answer_correctness_cb")

            # 收集选中的指标
            selected_metrics = []
            if faithfulness_check:
                selected_metrics.append('faithfulness')
            if answer_relevancy_check:
                selected_metrics.append('answer_relevancy')
            if context_precision_check:
                selected_metrics.append('context_precision')
            if context_recall_check:
                selected_metrics.append('context_recall')
            if answer_similarity_check:
                selected_metrics.append('answer_similarity')
            if answer_correctness_check:
                selected_metrics.append('answer_correctness')

            st.markdown(f"**已选择 {len(selected_metrics)} 个指标**")
            
            # 显示评估预估信息
            if selected_metrics:
                sample_count = len(st.session_state.current_dataset)
                estimated_time = sample_count * len(selected_metrics) * 2  # 每个指标每个样本约2秒
                st.info(f"📊 将评估 {sample_count} 个样本 × {len(selected_metrics)} 个指标 = {sample_count * len(selected_metrics)} 次评估")
              


            # 开始评估
            if st.button("🎯 开始评估", key="start_evaluation", type="primary"):
                if selected_metrics:
                    with st.container():
                        st.markdown("---")
                        st.markdown("### 🚀 评估进行中")
                        
                        results = evaluate_dataset(st.session_state.current_dataset, selected_metrics)
                        if results:
                            st.session_state.evaluation_results = results
                            
                            st.success("🎉 评估完成！请查看结果页面")
                            st.info("👉 点击上方 '📊 结果查看' 标签页查看详细结果")
                else:
                    st.warning("请至少选择一个评估指标")
        else:
            st.info("请先在数据集生成页面生成或处理数据集")

    with tab3:
        st.header("📊 评估结果")

        if st.session_state.evaluation_results and st.session_state.current_dataset:
            display_evaluation_results(st.session_state.evaluation_results, st.session_state.current_dataset)

            # 导出结果
            if st.button("💾 导出评估结果"):
                if hasattr(st.session_state, 'detailed_results_df'):
                    # 参考evaluate_dataset.py，但针对Streamlit下载进行优化
                    csv_buffer = st.session_state.detailed_results_df.to_csv(index=False, encoding='utf-8-sig')
                    # 转换为bytes格式，确保BOM正确传递
                    csv_bytes = csv_buffer.encode('utf-8-sig')

                    st.download_button(
                        label="📄 下载CSV格式详细结果",
                        data=csv_bytes,
                        file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("没有可下载的详细结果数据")
        else:
            st.info("暂无评估结果，请先在评估配置页面完成评估")

    # 底部日志显示
    if st.session_state.processing_log:
        with st.expander("📝 处理日志"):
            for log in st.session_state.processing_log[-10:]:  # 显示最近10条
                st.text(log)

if __name__ == "__main__":
    main()