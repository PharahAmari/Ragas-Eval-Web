#!/usr/bin/env python3
"""
并行版本的获取AI回答脚本
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
import json
from datetime import datetime
import logging

# API配置
API_URL = "https://aiagent-server.x.digitalyili.com/oapi/assistant/v1/session/run/4bcb486f-86e5-4502-a40b-fdfd976452ce"
headers = {
    'Authorization': 'Bearer AP_O0QsG1Bhfvz5PSou',
    'Content-Type': 'application/json'
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程锁，用于安全的文件操作
file_lock = threading.Lock()

def parse_reference_names(response_text):
    """从API响应文本中解析出reference的名字"""
    reference_files = []
    
    if not response_text:
        return reference_files
    
    # 查找所有docFileName字段
    if 'docFileName' in response_text:
        # 使用正则表达式提取文件名 - 匹配URL编码的docFileName字段
        file_matches = re.findall(r'docFileName%22%3A%22([^%22]+\.doc)%22', response_text)
        
        for file_match in file_matches:
            # URL解码文件名
            decoded_filename = urllib.parse.unquote(file_match)
            # 去重添加到列表
            if decoded_filename not in reference_files:
                reference_files.append(decoded_filename)
    
    # 也尝试匹配其他可能的文件名格式
    unencoded_matches = re.findall(r'"docFileName":"([^"]+\.doc)"', response_text)
    for match in unencoded_matches:
        if match not in reference_files:
            reference_files.append(match)
    
    # 匹配其他文档格式
    xlsx_matches = re.findall(r'docFileName%22%3A%22([^%22]+\.xlsx)%22', response_text)
    for match in xlsx_matches:
        decoded_filename = urllib.parse.unquote(match)
        if decoded_filename not in reference_files:
            reference_files.append(decoded_filename)
    
    return reference_files

def clean_ai_response(response_text):
    """清理AI响应，提取纯文本答案"""
    if not response_text:
        return ""
    
    try:
        # 尝试从响应中提取主要内容
        if '"content":{"type":"text","value":"' in response_text:
            start = response_text.find('"content":{"type":"text","value":"') + len('"content":{"type":"text","value":"')
            end = response_text.find('"}', start)
            if end != -1:
                answer = response_text[start:end]
                # 简单的JSON转义处理
                answer = answer.replace('\\"', '"').replace('\\n', '\n')
                
                # 首先提取主要回答部分（在div标签之前）
                main_answer = answer
                if '<div id="referenceSource"' in answer:
                    main_answer = answer.split('<div id="referenceSource"')[0].strip()
                
                return main_answer.strip()
        
        return "无法解析回答内容"
        
    except Exception as e:
        logger.error(f"清理响应时出错: {e}")
        return "响应解析出错"

def query_answer(question, thread_id):
    """发送单个问题并获取AI回答"""
    payload = {
        "thirdId": f"并行回答-{thread_id}",
        "question": {"type": "text", "value": question},
        "startFlowId": "ac343be8-1dc2-4758-88c8-70ad4456191d",
        "startNodeId": "01K35KW8BNYJ2HMW8E8S75RMRV"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        response_text = response.text
        logger.info(f"[线程{thread_id}] 获取回答成功: {question[:30]}...")
        
        # 尝试从响应中提取AI回答和参考信息
        if '"content":{"type":"text","value":"' in response_text:
            start = response_text.find('"content":{"type":"text","value":"') + len('"content":{"type":"text","value":"')
            end = response_text.find('"}', start)
            if end != -1:
                answer = response_text[start:end]
                # 简单的JSON转义处理
                answer = answer.replace('\\"', '"').replace('\\n', '\n')
                
                # 首先提取主要回答部分（在div标签之前）
                ai_answer = answer
                if '<div id="referenceSource"' in answer:
                    ai_answer = answer.split('<div id="referenceSource"')[0].strip()
                
                # 提取并解码URL编码的参考信息
                reference_text = ""
                if '<div id="referenceSource"' in answer:
                    # 找到div标签内的URL编码内容
                    div_start = answer.find('<div id="referenceSource"')
                    div_content = answer[div_start:]
                    
                    # 提取URL编码的JSON数据
                    url_encoded_matches = re.findall(r'%[0-9A-Fa-f]{2}[^<]*', div_content)
                    if url_encoded_matches:
                        # 解码最长的URL编码片段
                        longest_encoded = max(url_encoded_matches, key=len)
                        try:
                            decoded_json = urllib.parse.unquote(longest_encoded)
                            # 如果解码后是JSON，尝试解析
                            if decoded_json.startswith('[') and '"docFileName"' in decoded_json:
                                import json
                                try:
                                    reference_data = json.loads(decoded_json)
                                    for ref in reference_data:
                                        if 'docFileName' in ref:
                                            if 'score' in ref:
                                                reference_text += f"- {ref['docFileName']} (相关度: {ref['score']})\n"
                                            else:
                                                reference_text += f"- {ref['docFileName']}\n"
                                except json.JSONDecodeError:
                                    # 如果JSON解析失败，使用正则提取文件名
                                    file_names = re.findall(r'"docFileName":"([^"]+)"', decoded_json)
                                    for file_name in file_names:
                                        reference_text += f"- {file_name}\n"
                        except Exception as e:
                            logger.warning(f"解码参考信息时出错: {e}")
            else:
                ai_answer = "无法解析回答内容"
                reference_text = ""
        else:
            ai_answer = "无法解析回答内容"  
            reference_text = ""
        
        return {
            'ai_answer': ai_answer,
            'reference': reference_text,
            'success': True,
            'error': None
        }
        
    except requests.exceptions.Timeout:
        error_msg = "请求超时"
        logger.error(f"[线程{thread_id}] {error_msg}: {question[:30]}...")
        return {'ai_answer': "", 'reference': "", 'success': False, 'error': error_msg}
        
    except requests.exceptions.RequestException as e:
        error_msg = f"请求错误: {str(e)}"
        logger.error(f"[线程{thread_id}] {error_msg}: {question[:30]}...")
        return {'ai_answer': "", 'reference': "", 'success': False, 'error': error_msg}
        
    except Exception as e:
        error_msg = f"处理错误: {str(e)}"
        logger.error(f"[线程{thread_id}] {error_msg}: {question[:30]}...")
        return {'ai_answer': "", 'reference': "", 'success': False, 'error': error_msg}

def safe_update_excel(file_path, index, ai_answer, reference):
    """安全地更新Excel文件中的单行数据"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with file_lock:
                # 重新读取最新的文件状态
                df = pd.read_excel(file_path)
                
                # 更新指定行的数据
                df.at[index, 'AI回答'] = ai_answer
                df.at[index, '参考文档'] = reference
                
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
    
    # 检查是否已有AI回答
    if not pd.isna(row['AI回答']) and str(row['AI回答']).strip() != "":
        logger.info(f"[线程{thread_id}] 问题 {index + 1} 已有回答，跳过")
        return {'index': index, 'status': 'skipped', 'question': question}
    
    # 获取AI回答
    result = query_answer(question, thread_id)
    
    if result['success']:
        # 安全地更新文件
        if safe_update_excel(file_path, index, result['ai_answer'], result['reference']):
            logger.info(f"[线程{thread_id}] 问题 {index + 1} 处理完成并保存")
            return {'index': index, 'status': 'completed', 'question': question}
        else:
            logger.error(f"[线程{thread_id}] 问题 {index + 1} 保存失败")
            return {'index': index, 'status': 'save_failed', 'question': question}
    else:
        logger.error(f"[线程{thread_id}] 问题 {index + 1} 获取回答失败: {result['error']}")
        return {'index': index, 'status': 'api_failed', 'question': question, 'error': result['error']}

def process_file_parallel(excel_file, max_workers=4):
    """并行处理Excel文件中的所有问题"""
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        logger.info(f"读取到 {len(df)} 条记录")
        
        # 确保必要的列存在
        if 'AI回答' not in df.columns:
            df['AI回答'] = ''
        if '参考文档' not in df.columns:
            df['参考文档'] = ''
        df['AI回答'] = df['AI回答'].astype('object')
        df['参考文档'] = df['参考文档'].astype('object')
        
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
        print("用法: python get_answer_parallel.py <Excel文件路径> [线程数]")
        print("示例: python get_answer_parallel.py /path/to/your/file.xlsx 8")
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
    
    print(f"开始并行处理AI回答: {file_path}")
    print(f"使用线程数: {max_workers}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    process_file_parallel(file_path, max_workers)
    end_time = time.time()
    
    print("=" * 60)
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")