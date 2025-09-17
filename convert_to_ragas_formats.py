#!/usr/bin/env python3
"""
创建两种Ragas评估格式的数据集
方案1: 使用AI回答列作为answer
方案2: 让Ragas自己生成答案
"""

import pandas as pd
import json
import re
from typing import List, Dict
import sys

def extract_contexts_from_contexts_column(contexts_text: str) -> List[str]:
    """从Contexts列中提取chunk内容作为contexts列表"""
    if pd.isna(contexts_text) or not contexts_text:
        return ["无参考文档"]
    
    contexts = []
    contexts_text = str(contexts_text).strip()
    
    # 按文件标题分割chunks: [文件名]: 内容
    # 使用正则表达式匹配文件标题格式
    chunk_pattern = r'\[([^\]]+\.\w+)\]:\s*\n([^[]*?)(?=\n\[[^\]]+\.\w+\]:|$)'
    matches = re.findall(chunk_pattern, contexts_text, re.DOTALL)
    
    if matches:
        for filename, content in matches:
            # 清理内容：去掉多余空白和换行
            cleaned_content = re.sub(r'\s+', ' ', content.strip())
            if cleaned_content:
                contexts.append(cleaned_content)
    else:
        # 如果没有匹配到标准格式，尝试按双换行分割
        chunks = contexts_text.split('\n\n')
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and len(chunk) > 10:  # 过滤太短的内容
                # 移除可能的文件标题标记
                chunk = re.sub(r'^\[.*?\]:\s*', '', chunk)
                contexts.append(chunk)
    
    return contexts if contexts else ["参考文档信息不明确"]

def extract_contexts_from_reference(reference_text: str) -> List[str]:
    """从参考文档中提取真实的contexts"""
    if pd.isna(reference_text) or not reference_text:
        return ["无参考文档"]
    
    contexts = []
    
    # 提取文档名称（相关度最高的前3个）
    doc_pattern = r'- (.+?\.(?:xlsx|doc|pdf)) \(相关度: ([\d.]+)\)'
    matches = re.findall(doc_pattern, reference_text)
    
    if matches:
        # 按相关度排序，取前3个
        sorted_matches = sorted(matches, key=lambda x: float(x[1]), reverse=True)[:3]
        for doc, score in sorted_matches:
            contexts.append(f"来源文档: {doc}")
    
    # 如果没有找到文档信息，使用原始文本的前几行
    if not contexts:
        lines = reference_text.strip().split('\n')
        contexts = [line.strip() for line in lines[:2] if line.strip() and '相关度' not in line][:3]
    
    return contexts if contexts else ["参考文档信息不明确"]

def clean_ai_answer(ai_answer: str) -> str:
    """清理AI回答，删除图片链接但保留文字内容"""
    if pd.isna(ai_answer) or not ai_answer:
        return ""
    
    ai_answer = str(ai_answer).strip()
    
    # 删除图片相关的标签和链接，但保留其他文字内容
    # 1. 删除<img>标签
    ai_answer = re.sub(r'<img[^>]*>', '', ai_answer, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. 删除"相关图片"段落（包括各种可能的格式）
    # 匹配 "\n\n相关图片xxx\n\n" 的段落
    ai_answer = re.sub(r'\n\n\s*相关图片[^.\n]*?\s*\n\n', '\n\n', ai_answer, flags=re.IGNORECASE)
    # 匹配结尾的相关图片段落
    ai_answer = re.sub(r'\n\n\s*相关图片[^.\n]*?\s*$', '', ai_answer, flags=re.IGNORECASE)
    # 匹配单独一行的相关图片
    ai_answer = re.sub(r'\n\s*相关图片[^.\n]*?\s*\n', '\n', ai_answer, flags=re.IGNORECASE)
    # 匹配行末的相关图片
    ai_answer = re.sub(r'相关图片[^.\n]*?$', '', ai_answer, flags=re.IGNORECASE | re.MULTILINE)
    
    # 3. 清理多余的空行
    ai_answer = re.sub(r'\n\s*\n\s*\n', '\n\n', ai_answer)
    
    # 4. 清理首尾空白
    ai_answer = ai_answer.strip()
    
    return ai_answer

def create_format1_ai_answer(excel_path: str, output_path: str) -> None:
    """
    方案1: 使用AI回答列作为answer
    - question: 问题
    - answer: AI回答列的内容（被评估的回答）
    - contexts: Contexts列中的chunk内容（作为参考上下文）
    - ground_truth: 标准答案（评估基准）
    """
    
    print("🔄 创建方案1: AI回答作为answer，Contexts列作为contexts...")
    df = pd.read_excel(excel_path)
    
    ragas_data = []
    
    for idx, row in df.iterrows():
        # 跳过没有必要字段的行
        if pd.isna(row['问题']) or pd.isna(row['标准答案']) or pd.isna(row['AI回答']):
            continue
        
        # 从Contexts列提取chunk内容作为contexts
        if 'Contexts' in row and not pd.isna(row['Contexts']):
            contexts = extract_contexts_from_contexts_column(row['Contexts'])
        else:
            # 如果没有Contexts列，回退到使用标准答案
            standard_answer = str(row['标准答案']).strip()
            if len(standard_answer) > 200:
                # 长答案分段作为多个contexts
                segments = re.split(r'[；;。]', standard_answer)
                contexts = [seg.strip() for seg in segments[:3] if seg.strip()]
            else:
                contexts = [standard_answer]
        
        # 清理AI回答，删除图片相关内容后面的部分
        cleaned_ai_answer = clean_ai_answer(row['AI回答'])
        
        item = {
            "question": str(row['问题']).strip(),
            "answer": cleaned_ai_answer,  # 清理后的AI回答
            "contexts": contexts,  # Contexts列的chunk内容作为上下文
            "ground_truth": str(row['标准答案']).strip(),  # 标准答案
        }
        
        ragas_data.append(item)
    
    # 保存数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 方案1完成，共 {len(ragas_data)} 条数据")
    print(f"💾 已保存到: {output_path}")
    
    # 显示样例
    if ragas_data:
        sample = ragas_data[0]
        print(f"\n📝 方案1样例:")
        print(f"问题: {sample['question'][:30]}...")
        print(f"AI回答: {sample['answer'][:50]}...")
        print(f"标准答案: {sample['ground_truth'][:50]}...")
        print(f"Contexts数量: {len(sample['contexts'])}")
        print(f"Contexts示例: {sample['contexts'][0][:100] if sample['contexts'] else '无'}...")

def create_format2_empty_answer(excel_path: str, output_path: str) -> None:
    """
    方案2: 让Ragas自己生成答案
    - question: 问题
    - answer: 空字符串（让Ragas的LLM生成）
    - contexts: Contexts列中的chunk内容（作为参考上下文）
    - ground_truth: 标准答案（评估基准）
    """
    
    print("🔄 创建方案2: 让Ragas生成答案，Contexts列作为contexts...")
    df = pd.read_excel(excel_path)
    
    ragas_data = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['问题']) or pd.isna(row['标准答案']):
            continue
        
        # 从Contexts列提取chunk内容作为contexts
        if 'Contexts' in row and not pd.isna(row['Contexts']):
            contexts = extract_contexts_from_contexts_column(row['Contexts'])
        else:
            # 如果没有Contexts列，回退到使用标准答案
            standard_answer = str(row['标准答案']).strip()
            if len(standard_answer) > 200:
                # 长答案分段作为多个contexts
                segments = re.split(r'[；;。]', standard_answer)
                contexts = [seg.strip() for seg in segments[:3] if seg.strip()]
            else:
                contexts = [standard_answer]
        
        item = {
            "question": str(row['问题']).strip(),
            "answer": "",  # 空答案，让Ragas生成
            "contexts": contexts,  # Contexts列的chunk内容作为上下文
            "ground_truth": str(row['标准答案']).strip(),  # 标准答案
        }
        
        ragas_data.append(item)
    
    # 保存数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 方案2完成，共 {len(ragas_data)} 条数据")
    print(f"💾 已保存到: {output_path}")
    
    # 显示样例
    if ragas_data:
        sample = ragas_data[0]
        print(f"\n📝 方案2样例:")
        print(f"问题: {sample['question'][:30]}...")
        print(f"AI回答: [空，待生成]")
        print(f"标准答案: {sample['ground_truth'][:50]}...")
        print(f"Contexts数量: {len(sample['contexts'])}")
        print(f"Contexts示例: {sample['contexts'][0][:100] if sample['contexts'] else '无'}...")

def main():
    excel_file = sys.argv[1]
    
    # 方案1: 评估现有AI回答
    format1_output = sys.argv[2]
    create_format1_ai_answer(excel_file, format1_output)
    
    # print("\n" + "="*60 + "\n")
    
    # # 方案2: 让Ragas LLM生成新回答
    # format2_output = sys.argv[2]
    # create_format2_empty_answer(excel_file, format2_output)
    
    # print(f"\n🎯 两种格式都已创建完成!")
    # print(f"📁 评估现有AI回答: {format1_output}")
    # print(f"📁 评估LLM生成回答: {format2_output}")

if __name__ == "__main__":
    main()