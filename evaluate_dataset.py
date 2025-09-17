#!/usr/bin/env python3

import json
import pandas as pd
import os
import sys
import time
import logging
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random

class DatasetEvaluator:
    def __init__(self, request_timeout: int = 600):
        # 配置日志
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('evaluation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 检查API key
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            self.logger.error("DASHSCOPE_API_KEY 环境变量未设置")
            sys.exit(1)
        
        self.logger.info(f"API Key: {api_key[:10]}...")
        
        # 配置LLM - 增加超时时间以适应梯子网络
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_retries=5,  # 增加重试次数
            request_timeout=request_timeout,  # 可配置超时时间
            max_tokens=8096,
            temperature=0.0,
        )
        
        # 配置embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            cache_folder="/home/zhangdh17/.cache/huggingface/hub/"
        )
        
        # 选择评估指标
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        ]
        
        self.logger.info(f"使用评估指标: {[m.name for m in self.metrics]}")
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """加载评估数据集"""
        self.logger.info(f"加载数据集: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            self.logger.error(f"数据集文件不存在: {dataset_path}")
            sys.exit(1)
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"成功加载 {len(data)} 个样本")
        return data
    
    def create_ragas_dataset(self, evaluation_data: List[Dict]) -> Dataset:
        """创建Ragas格式的数据集"""
        self.logger.info("创建Ragas数据集...")
        
        questions = [item['question'] for item in evaluation_data]
        answers = [item['answer'] for item in evaluation_data]
        contexts = [item['contexts'] for item in evaluation_data]
        ground_truths = [item.get('ground_truth', item['answer']) for item in evaluation_data]
        
        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
    
    def run_evaluation(self, dataset: Dataset, max_retries: int = 3):
        """运行一次性评估，带重试机制"""
        self.logger.info(f"开始评估，样本数: {len(dataset)}")
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"评估尝试 {attempt + 1}/{max_retries}")
                
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False,  # 改为False以获得更好的错误处理
                )
                
                end_time = time.time()
                duration = end_time - start_time
                self.logger.info(f"评估完成，耗时: {duration:.2f}秒")
                
                # 检查评估结果质量
                self._check_evaluation_quality(result)
                
                return result
                
            except Exception as e:
                self.logger.error(f"评估尝试 {attempt + 1} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 递增等待时间
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("所有评估尝试都失败了")
                    raise e
    
    def _check_evaluation_quality(self, result):
        """检查评估结果质量"""
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            
            for metric in self.metrics:
                if metric.name in df.columns:
                    col_data = df[metric.name]
                    zero_count = (col_data == 0).sum()
                    nan_count = col_data.isna().sum()
                    
                    if zero_count > 0:
                        self.logger.warning(f"{metric.name}: {zero_count}个零值 (可能表示该指标评估失败)")
                    if nan_count > 0:
                        self.logger.warning(f"{metric.name}: {nan_count}个空值 (评估未完成)")
                    
                    self.logger.info(f"{metric.name}: 平均分={col_data.mean():.4f}, 范围=[{col_data.min():.4f}, {col_data.max():.4f}]")
    
    def run_batch_evaluation(self, dataset: Dataset, batch_size: int = 10):
        """运行批次评估，适合大型数据集"""
        self.logger.info(f"开始批次评估，样本数: {len(dataset)}, 批次大小: {batch_size}")
        
        total_samples = len(dataset)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        all_results = []
        successful_batches = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            
            # 创建批次数据集
            batch_data = {
                "question": dataset["question"][start_idx:end_idx],
                "answer": dataset["answer"][start_idx:end_idx],
                "contexts": dataset["contexts"][start_idx:end_idx],
                "ground_truth": dataset["ground_truth"][start_idx:end_idx]
            }
            batch_dataset = Dataset.from_dict(batch_data)
            
            self.logger.info(f"处理批次 {i+1}/{num_batches} (样本 {start_idx+1}-{end_idx})")
            
            result = self._evaluate_batch(batch_dataset, i+1)
            if result is not None:
                all_results.append(result)
                successful_batches += 1
            else:
                self.logger.warning(f"批次 {i+1} 评估失败，跳过")
        
        if not all_results:
            self.logger.error("所有批次评估都失败了")
            return None
        
        self.logger.info(f"成功评估 {successful_batches}/{num_batches} 个批次")
        return self._merge_results(all_results)
    
    def _evaluate_batch(self, batch_dataset: Dataset, batch_num: int, max_retries: int = 3):
        """评估单个批次，增强重试机制和限流检测"""
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"批次 {batch_num} 尝试 {attempt + 1}/{max_retries}")
                
                # 添加随机延迟来避免同步请求
                if batch_num > 1:
                    jitter = random.uniform(0.5, 2.0)
                    time.sleep(jitter)
                    self.logger.info(f"批次 {batch_num} 随机延迟 {jitter:.1f}s")
                
                start_time = time.time()
                
                result = evaluate(
                    dataset=batch_dataset,
                    metrics=self.metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                    batch_size=1,  # 保持小批次大小
                    raise_exceptions=False,
                )
                
                eval_time = time.time() - start_time
                self.logger.info(f"批次 {batch_num} 评估成功，耗时: {eval_time:.1f}s")
                return result
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"批次 {batch_num} 尝试 {attempt + 1} 失败: {error_msg}")
                
                if attempt < max_retries - 1:
                    # 智能等待时间计算
                    if "Connection error" in error_msg or "timeout" in error_msg.lower():
                        # API限流检测到，使用更长的等待时间
                        wait_time = base_delay * (3 ** attempt) + random.uniform(5, 15)
                        self.logger.warning(f"检测到API限流，延长等待时间至 {wait_time:.1f}s")
                    elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                        # 明确的限流错误
                        wait_time = base_delay * (4 ** attempt) + random.uniform(10, 30)
                        self.logger.warning(f"明确的限流错误，等待 {wait_time:.1f}s")
                    else:
                        wait_time = base_delay + random.uniform(1, 5)
                    
                    self.logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
        
        self.logger.error(f"批次 {batch_num} 所有重试都失败了")
        return None
    
    def _merge_results(self, results_list):
        """合并多个评估结果"""
        # 获取第一个结果作为基础
        merged_result = results_list[0]
        
        if len(results_list) == 1:
            return merged_result
        
        # 如果有多个结果，这里可以实现更复杂的合并逻辑
        self.logger.warning("使用简单合并策略，返回第一个批次结果")
        return merged_result
    
    def save_results(self, result, evaluation_data: List[Dict], output_path: str = "evaluation_results.csv"):
        """保存评估结果"""
        self.logger.info("保存评估结果...")
        
        # 创建基础DataFrame
        df_data = {
            'question': [item['question'] for item in evaluation_data],
            'answer': [item['answer'] for item in evaluation_data],
            'ground_truth': [item.get('ground_truth', item['answer']) for item in evaluation_data]
        }
        
        # 添加评估分数
        if hasattr(result, 'to_pandas'):
            scores_df = result.to_pandas()
            # 合并数据
            for col in scores_df.columns:
                if col not in df_data:
                    df_data[col] = scores_df[col].tolist()
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 添加BOM标记，Windows兼容
        
        self.logger.info(f"结果已保存到: {output_path}")
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """打印评估结果摘要"""
        print("\n" + "="*50)
        print("评估结果摘要")
        print("="*50)
        
        # 找到数值列
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        if len(numeric_columns) == 0:
            print("未找到数值评估结果")
            return
        
        for col in numeric_columns:
            if col in df.columns:
                avg_score = df[col].mean()
                min_score = df[col].min()
                max_score = df[col].max()
                print(f"{col}:")
                print(f"  平均分: {avg_score:.4f}")
                print(f"  最低分: {min_score:.4f}")
                print(f"  最高分: {max_score:.4f}")
        
        # 总体评分
        if len(numeric_columns) > 0:
            overall_score = df[numeric_columns].mean().mean()
            print(f"\n总体评分: {overall_score:.4f}")
        
        print("="*50)

def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python evaluate_dataset.py <数据集文件>")
        print("示例: python evaluate_dataset.py EvalDataset_format1_ai_answer_optimized.json")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    
    print(f"开始评估数据集: {dataset_file}")
    print("-" * 50)
    
    # 创建评估器
    evaluator = DatasetEvaluator()
    
    try:
        # 加载数据
        evaluation_data = evaluator.load_dataset(dataset_file)
        
        # 创建数据集
        dataset = evaluator.create_ragas_dataset(evaluation_data)
        
        # 直接评估，不使用批处理
        dataset_size = len(dataset)
        print(f"数据集大小: {dataset_size}个样本，使用直接评估模式")
        result = evaluator.run_evaluation(dataset)
        
        if result is None:
            print("评估失败")
            sys.exit(1)
        
        # 保存结果
        df = evaluator.save_results(result, evaluation_data)
        
        # 打印摘要
        evaluator.print_summary(df)
        
        print("\n评估完成!")
        
    except Exception as e:
        evaluator.logger.error(f"评估过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()