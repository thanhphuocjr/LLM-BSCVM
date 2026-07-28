
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modules.vulnerability_detector import VulnerabilityDetector
from config.config import Config

class SmartContractAnalyzer:
    """智能合约分析器主类 - 仅漏洞检测版本"""
    
    def __init__(self):
        """初始化分析器组件"""
        self.config = Config()
        self._setup_logging()
        
        self.logger.info("初始化漏洞检测器...")
        # 仅初始化漏洞检测器，只加载一次模型
        self.vulnerability_detector = VulnerabilityDetector(
            self.config.FINETUNED_MODEL_PATH,
            self.config.KNOWLEDGE_BASE_PATH
        )
        self.logger.info("漏洞检测器初始化完成")

    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("output/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'detection_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_single_contract(self, contract_code: str) -> Dict[str, Any]:
        """分析单个合约 - 仅执行漏洞检测"""
        try:
            # 仅执行漏洞检测任务，不输出日志
            return self.vulnerability_detector.analyze(contract_code)
        except Exception as e:
            self.logger.error(f"合约分析过程中发生错误: {str(e)}")
            raise

    def _save_detailed_evaluation(self, contract_data: dict, analysis_results: dict) -> dict:
        """记录漏洞检测的评估结果"""
        if not isinstance(analysis_results, dict):
            analysis_results = {'vulnerability_check': analysis_results}
        
        return {
            "contract_info": {
                "id": contract_data.get("id", "unknown"),
                "original_code": contract_data.get("code", ""),
                "ground_truth_label": contract_data.get("ground_truth_label", "")
            },
            "detection_evaluation": {
                "predicted": analysis_results.get("vulnerability_check", ""),
                "ground_truth": contract_data.get("ground_truth_label", ""),
                "is_correct": str(analysis_results.get("vulnerability_check", "")).lower() == 
                            str(contract_data.get("ground_truth_label", "")).lower()
            }
        }

    def analyze_dataset(self, dataset_path: str = None):
        """分析整个数据集 - 仅执行漏洞检测"""
        if dataset_path is None:
            dataset_path = self.config.VALID_DATA_PATH
            
        try:
            # 加载数据集
            with open(dataset_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            self.logger.info(f"成功加载测试数据，共 {len(test_data)} 条记录")
            
            # 创建输出目录
            results_dir = Path("output/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 存储评估结果
            detailed_evaluations = []
            true_labels = []
            predicted_labels = []
            
            # 记录检测任务的统计信息
            detection_statistics = {
                "correct": 0,
                "total": 0,
                "success_rate": "0.00%"
            }
            
            # 处理每个测试用例
            for idx, contract_data in enumerate(tqdm(test_data, desc="检测进度", ncols=100)):
                try:
                    if not isinstance(contract_data, dict):
                        continue
                    
                    contract_id = contract_data.get("id", f"unknown_{idx}")
                    contract_code = contract_data.get("code", "")
                    
                    if not contract_code:
                        self.logger.warning(f"合约 {contract_id} 没有代码内容，跳过")
                        continue
                    
                    # 分析合约
                    analysis_results = self.analyze_single_contract(contract_code)
                    if not isinstance(analysis_results, dict):
                        raise ValueError(f"分析结果格式错误: {type(analysis_results)}")
                    
                    # 保存评估结果
                    detailed_eval = self._save_detailed_evaluation(contract_data, analysis_results)
                    detailed_evaluations.append(detailed_eval)
                    
                    # 更新检测统计
                    detection_statistics["total"] += 1
                    if detailed_eval["detection_evaluation"]["is_correct"]:
                        detection_statistics["correct"] += 1
                    
                    # 收集评估指标数据
                    if (detailed_eval["detection_evaluation"]["ground_truth"] and 
                        detailed_eval["detection_evaluation"]["predicted"]):
                        true_labels.append(detailed_eval["detection_evaluation"]["ground_truth"].lower())
                        predicted_labels.append(detailed_eval["detection_evaluation"]["predicted"].lower())
                    
                except Exception as e:
                    self.logger.error(f"处理合约 {contract_id} 时发生错误: {str(e)}")
                    self.logger.exception("详细错误信息:")
                    continue

            # 计算评估指标
            classification_metrics = {}
            if true_labels and predicted_labels:
                classification_metrics = {
                    'accuracy': accuracy_score(true_labels, predicted_labels),
                    'precision': precision_score(true_labels, predicted_labels, pos_label='vulnerable'),
                    'recall': recall_score(true_labels, predicted_labels, pos_label='vulnerable'),
                    'f1': f1_score(true_labels, predicted_labels, pos_label='vulnerable')
                }

            # 计算成功率
            if detection_statistics["total"] > 0:
                detection_statistics["success_rate"] = f"{(detection_statistics['correct'] / detection_statistics['total']) * 100:.2f}%"
            
            # 准备最终评估结果
            final_evaluation = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_contracts": len(test_data),
                "processed_contracts": len(detailed_evaluations),
                "classification_metrics": classification_metrics,
                "detection_statistics": detection_statistics,
                "detailed_evaluations": detailed_evaluations
            }
            
            # 保存最终评估结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_results_file = results_dir / f'detection_evaluation_results_{timestamp}.json'
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(final_evaluation, f, indent=2, ensure_ascii=False)
            
            # 打印评估摘要
            self.logger.info("\n=== 漏洞检测评估结果汇总 ===")
            self.logger.info(f"总样本数: {len(test_data)}")
            self.logger.info(f"成功处理样本数: {len(detailed_evaluations)}")
            
            if classification_metrics:
                self.logger.info("\n分类指标:")
                for metric, value in classification_metrics.items():
                    self.logger.info(f"{metric}: {value:.4f}")
            
            self.logger.info(f"\n检测统计:")
            for metric, value in detection_statistics.items():
                self.logger.info(f"{metric}: {value}")
            
            self.logger.info(f"\n详细评估结果已保存至: {final_results_file}")
            
        except Exception as e:
            self.logger.error(f"数据集处理过程中发生错误: {str(e)}")
            self.logger.exception("详细错误信息:")
            raise

def main():
    """主函数"""
    try:
        analyzer = SmartContractAnalyzer()
        analyzer.analyze_dataset()
        
    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
