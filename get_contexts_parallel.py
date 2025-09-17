#!/usr/bin/env python3
"""
并行版本的获取上下文脚本
- 使用多线程同时处理多个问题
- 避免文件冲突的安全写入机制
- 支持断点续传和错误重试
"""

import requests
import pandas as pd
import time
import re
import urllib.parse
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
import html2text
from datetime import datetime
import logging

# API配置 - 使用contexts retrieve的API
API_URL = "https://aiagent-server.x.digitalyili.com/oapi/assistant/v1/session/run/95663b0c-5655-44c0-a6cd-24b3a1de29c6"
headers = {
    'Authorization': 'Bearer AP_O0QsG1Bhfvz5PSou',
    'Content-Type': 'application/json'
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程锁，用于安全的文件操作
file_lock = threading.Lock()

def html_to_text(html_content):
    """将HTML格式的内容转换为纯文本"""
    if not html_content or not isinstance(html_content, str):
        return html_content
    
    try:
        # 使用html2text库转换HTML到文本
        h = html2text.HTML2Text()
        h.ignore_links = False  # 保留链接
        h.ignore_images = True  # 忽略图片
        h.ignore_emphasis = False  # 保留强调格式
        h.body_width = 0  # 不限制行宽
        
        # 先进行HTML解码
        decoded_content = unescape(html_content)
        
        # 转换HTML到Markdown格式的文本
        text_content = h.handle(decoded_content)
        
        # 清理多余的空行
        text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content)
        text_content = text_content.strip()
        
        # 处理chunk文件标题格式，让标题顶格，内容另起一行
        # 匹配模式：[文件名]: 后面跟内容
        text_content = re.sub(r'(\[.*?\.\w+\]):\s*```\s*', r'\1:\n', text_content)
        text_content = re.sub(r'(\[.*?\.\w+\]):\s*([^`\n])', r'\1:\n\2', text_content)
        
        # 清理chunk分隔符和多余的反引号
        text_content = re.sub(r'\s*```\s*—+\s*', '\n\n', text_content)
        text_content = re.sub(r'```\s*$', '', text_content, flags=re.MULTILINE)
        
        # 再次清理多余的空行
        text_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_content)
        
        return text_content
    except Exception as e:
        logger.error(f"HTML转换出错: {e}")
        # 如果转换失败，使用简单的正则表达式清理HTML标签
        try:
            # 移除HTML标签
            text_content = re.sub(r'<[^>]+>', '', html_content)
            # HTML解码
            text_content = unescape(text_content)
            # 清理多余空白
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # 处理chunk文件标题格式，让标题顶格，内容另起一行
            text_content = re.sub(r'(\[.*?\.\w+\]):\s*```\s*', r'\1:\n', text_content)
            text_content = re.sub(r'(\[.*?\.\w+\]):\s*([^`\n])', r'\1:\n\2', text_content)
            
            # 清理chunk分隔符和多余的反引号
            text_content = re.sub(r'\s*```\s*—+\s*', '\n\n', text_content)
            text_content = re.sub(r'```\s*$', '', text_content, flags=re.MULTILINE)
            
            return text_content
        except:
            return html_content

def query_contexts(question, thread_id):
    """向AI发送问题并获取上下文信息"""
    payload = {
        "thirdId": f"并行上下文-{thread_id}",
        "question": {"type": "text", "value": question},
        "startFlowId": "0b9ad9bc-9dbd-4c3b-8277-18ca36288fd0",
        "startNodeId": "01K4PWKW5F5R0SR5DHADK5SWEV"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # 简化响应处理，直接返回文本内容
        response_text = response.text
        logger.debug(f"[线程{thread_id}] 原始响应: {response_text[:200]}...")
        
        # 尝试从响应中提取上下文内容
        if '"content":{"type":"text","value":"' in response_text:
            start = response_text.find('"content":{"type":"text","value":"') + len('"content":{"type":"text","value":"')
            end = response_text.find('"}', start)
            if end != -1:
                contexts = response_text[start:end]
                # 简单的JSON转义处理
                contexts = contexts.replace('\\"', '"').replace('\\n', '\n')
                
                # 将HTML格式转换为文本格式
                contexts = html_to_text(contexts)
                
                logger.info(f"[线程{thread_id}] 获取上下文成功: {question[:30]}...")
                return contexts, True
        
        logger.warning(f"[线程{thread_id}] 无法解析上下文内容")
        return "", False
        
    except requests.exceptions.Timeout:
        logger.error(f"[线程{thread_id}] 请求超时: {question[:30]}...")
        return "", False
    except requests.exceptions.RequestException as e:
        logger.error(f"[线程{thread_id}] 请求错误: {str(e)}")
        return "", False
    except Exception as e:
        logger.error(f"[线程{thread_id}] 处理错误: {str(e)}")
        return "", False

def safe_update_excel(file_path, index, contexts):
    """安全地更新Excel文件中的单行数据"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with file_lock:
                # 重新读取最新的文件状态
                df = pd.read_excel(file_path)
                
                # 更新指定行的数据
                df.at[index, 'Contexts'] = contexts
                
                # 保存文件
                df.to_excel(file_path, index=False)
                return True
                
        except Exception as e:
            logger.warning(f"保存文件失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
            else:
                logger.error(f"保存文件最终失败: {e}")
                return False
    return False

def process_single_question(args):
    """处理单个问题的包装函数"""
    index, row, file_path, thread_id = args
    
    question = str(row['问题']).strip()
    logger.info(f"[线程{thread_id}] 开始处理问题 {index + 1}: {question[:50]}...")
    
    # 检查是否已有Contexts
    if not pd.isna(row['Contexts']) and str(row['Contexts']).strip() != "":
        logger.info(f"[线程{thread_id}] 问题 {index + 1} 已有上下文，跳过")
        return {'index': index, 'status': 'skipped', 'question': question}
    
    # 获取上下文内容
    contexts, is_success = query_contexts(question, thread_id)
    
    if is_success:
        # 安全地更新文件
        if safe_update_excel(file_path, index, contexts):
            logger.info(f"[线程{thread_id}] 问题 {index + 1} 处理完成并保存")
            return {'index': index, 'status': 'completed', 'question': question}
        else:
            logger.error(f"[线程{thread_id}] 问题 {index + 1} 保存失败")
            return {'index': index, 'status': 'save_failed', 'question': question}
    else:
        logger.error(f"[线程{thread_id}] 问题 {index + 1} 获取上下文失败")
        return {'index': index, 'status': 'api_failed', 'question': question}

def process_file_parallel(excel_file, max_workers=4):
    """并行处理Excel文件中的所有问题"""
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        logger.info(f"读取到 {len(df)} 条记录")
        
        # 确保Contexts列存在且为字符串类型
        if 'Contexts' not in df.columns:
            df['Contexts'] = ''
        df['Contexts'] = df['Contexts'].astype('object')
        
        # 初始保存，确保列存在
        df.to_excel(excel_file, index=False)
        
        # 准备任务参数
        tasks = []
        for index, row in df.iterrows():
            if pd.isna(row['问题']) or str(row['问题']).strip() == "":
                continue
            tasks.append((index, row, excel_file, None))  # thread_id会在执行时分配
        
        if not tasks:
            logger.warning("没有找到需要处理的问题")
            return
        
        logger.info(f"准备并行处理 {len(tasks)} 个问题，使用 {max_workers} 个线程")
        
        # 统计结果
        results = {
            'completed': 0,
            'skipped': 0,
            'api_failed': 0,
            'save_failed': 0
        }
        
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个任务分配thread_id
            tasks_with_id = [(task[0], task[1], task[2], i+1) for i, task in enumerate(tasks)]
            
            # 提交所有任务
            future_to_task = {executor.submit(process_single_question, task): task for task in tasks_with_id}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                result = future.result()
                results[result['status']] += 1
                
                # 每处理5个任务显示一次进度
                total_processed = sum(results.values())
                if total_processed % 5 == 0 or total_processed == len(tasks):
                    logger.info(f"进度: {total_processed}/{len(tasks)} "
                              f"(完成: {results['completed']}, 跳过: {results['skipped']}, "
                              f"API失败: {results['api_failed']}, 保存失败: {results['save_failed']})")
        
        # 最终统计
        logger.info("=" * 60)
        logger.info("处理完成！最终统计:")
        logger.info(f"总问题数: {len(tasks)}")
        logger.info(f"成功完成: {results['completed']}")
        logger.info(f"已存在跳过: {results['skipped']}")
        logger.info(f"API请求失败: {results['api_failed']}")
        logger.info(f"文件保存失败: {results['save_failed']}")
        logger.info(f"成功率: {results['completed']/(len(tasks)-results['skipped'])*100:.1f}%" if len(tasks) > results['skipped'] else "N/A")
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python get_contexts_parallel.py <Excel文件路径> [线程数]")
        print("示例: python get_contexts_parallel.py /path/to/your/file.xlsx 8")
        sys.exit(1)
    
    file_path = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)
    
    # 检查文件扩展名
    if not file_path.lower().endswith(('.xlsx', '.xls')):
        print("错误: 请提供Excel文件（.xlsx 或 .xls 格式）")
        sys.exit(1)
    
    print(f"开始并行处理上下文: {file_path}")
    print(f"使用线程数: {max_workers}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    process_file_parallel(file_path, max_workers)
    end_time = time.time()
    
    print("=" * 60)
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")