import re
from datetime import datetime
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
from modules.base_analyzer import BaseAnalyzer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import logging
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import Config


class LlamaAnalyzer:
    def __init__(self, model_path: str = Config.MODEL_PATH):
        """Initialize CodeLlama model"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Load model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"  # CodeLlama推荐设置
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
        
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
                
        except Exception as e:
            print(f"Error initializing CodeLlama model: {str(e)}")
            raise

    def get_response(self, prompt: str) -> Optional[str]:
        """Get response from CodeLlama model"""
        try:
            # 修改提示词格式以适应CodeLlama
            formatted_prompt = f"""[INST] <<SYS>> You are a Smart Contract Auditor and Solidity Expert. Your task is to analyze this smart contract feature and provide a very brief description focusing only on its core functionality.
<</SYS>>

{prompt}

[/INST]"""
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=Config.MODEL_CONFIG["context_length"]
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.MODEL_CONFIG["max_new_tokens"],
                    do_sample=True,
                    temperature=Config.MODEL_CONFIG["temperature"],
                    top_p=Config.MODEL_CONFIG["top_p"],
                    repetition_penalty=Config.MODEL_CONFIG["repetition_penalty"],
                    num_return_sequences=1
                )
            
            response = response.split('[/INST]')[-1].strip()
            
            sentences = re.split(r'[.!?]', response)  # 使用标点符号分割成句子
            sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]  # 去掉空白句子

            if len(sentences) > 0:
                response = '. '.join(sentences[:2]) + '.'  # 最多取前两句，加上句号
            else:
                response = "Unable to determine the function's core functionality."


            response = response[0].upper() + response[1:] if response else response

            return response

        except Exception as e:
            print(f"Error getting LLaMA response: {str(e)}")
            return None
        
class ReportGenerator(BaseAnalyzer):
    """Smart Contract Audit Report Generator"""
    
    def __init__(self):
        super().__init__()
        self.llm = LlamaAnalyzer()

    def _create_custom_styles(self):
        """Create custom styles for the report"""
        styles = getSampleStyleSheet()
        
        custom_styles = {
            'SectionTitle': {
                'parent': styles['Heading1'],
                'fontSize': 16,
                'spaceAfter': 12,
                'textColor': HexColor('#000000'),
                'fontName': 'Helvetica-Bold',
                'leftIndent': 0
            },
            'VulnerabilityTitle': {
            'parent': styles['Heading2'],
            'fontSize': 11,
            'spaceAfter': 6,
            'spaceBefore': 6,
            'textColor': HexColor('#000000'),
            'fontName': 'Helvetica-Bold'
            },
            'SubsectionTitle': {
                'parent': styles['Heading3'],
                'fontSize': 12,
                'spaceAfter': 8,
                'textColor': HexColor('#000000'),
                'fontName': 'Helvetica-Bold',
                'leftIndent': 40
            },
            'CustomNormal': {
            'parent': styles['Normal'],
            'fontSize': 10,
            'leading': 12,
            'spaceBefore': 2,
            'spaceAfter': 2,
            'textColor': HexColor('#000000'),
            'leftIndent': 20,
            'rightIndent': 20
            },
        
            'CodeStyle': {
                'parent': styles['Code'],
                'fontSize': 9,
                'fontName': 'Courier',
                'spaceAfter': 8,
                'textColor': HexColor('#000000'),
                'backColor': HexColor('#F5F5F5')
            },
            'CodeBlock': {
            'parent': styles['Code'],
            'fontSize': 9,
            'fontName': 'Courier',
            'spaceBefore': 4,
            'spaceAfter': 4,
            'leftIndent': 20,
            'rightIndent': 20,
            'backColor': HexColor('#F5F5F5'),
            'borderPadding': 4
            },
            'ReportTitle': {
                'parent': styles['Title'],
                'fontSize': 24,
                'spaceAfter': 10,
                'alignment': TA_LEFT,
                'textColor': HexColor('#000000'),
                'fontName': 'Helvetica-Bold',
                'leftIndent': 0
            },
            'ReportSubtitle': {
                'parent': styles['Title'],
                'fontSize': 18,
                'spaceAfter': 30,
                'alignment': TA_LEFT,
                'textColor': HexColor('#000000'),
                'fontName': 'Helvetica-Bold',
                'leftIndent': 0
            },
            'MainTitle': {
                'parent': styles['Heading1'],
                'fontSize': 16,
                'spaceAfter': 20,
                'textColor': HexColor('#000000'),
                'fontName': 'Helvetica-Bold'
            },
            'CustomHeading2': {
                'parent': styles['Heading2'],
                'fontSize': 14,
                'spaceAfter': 12,
                'textColor': HexColor('#000000'),
                'fontName': 'Helvetica-Bold'
            },
            'TableHeader': {
                'parent': styles['Normal'],
                'fontSize': 10,
                'textColor': HexColor('#FFFFFF'),
                'alignment': TA_CENTER
            }
        }
        
        # Only add styles that don't already exist
        for style_name, props in custom_styles.items():
            styles.add(ParagraphStyle(name=style_name, **props))
    
        return styles

            

    def _format_repair_section(self, repair_suggestion: str, styles) -> List:
        """Format repair suggestions for PDF report"""
        story = []

        # 直接显示修复建议文本，不需要额外处理
        if repair_suggestion == "No vulnerabilities were detected; therefore, no remediation actions are required.":
            story.append(Paragraph(repair_suggestion, styles['CustomNormal']))
            return story

        if not repair_suggestion:
            story.append(Paragraph("No repair suggestions available.", styles['CustomNormal']))
            return story

        try:
            # Split into vulnerability sections
            sections = repair_suggestion.split('\n\nVulnerability')
            vulnerability_count = 1

            for section in sections:
                if not section.strip():
                    continue

                # If section doesn't start with "Vulnerability", add it
                if not section.lower().startswith('vulnerability'):
                    section = 'Vulnerability' + section

                # Split into lines and process
                lines = section.split('\n')
                content = {
                    'name': '',
                    'description': '',
                    'impact': '',
                    'fix': '',
                    'prevention': ''
                }
                
                current_key = None
                
                # Process each line
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for section headers
                    if 'Name:' in line:
                        current_key = 'name'
                        content['name'] = line.split('Name:', 1)[1].strip()
                    elif line.startswith('Description:'):
                        current_key = 'description'
                        content['description'] = line.replace('Description:', '<b>Description:</b>')
                    elif line.startswith('Impact:'):
                        current_key = 'impact'
                        content['impact'] = line.replace('Impact:', '<b>Impact:</b>')
                    elif line.startswith('Fix:'):
                        current_key = 'fix'
                        content['fix'] = line.replace('Fix:', '<b>Fix:</b>')
                    elif line.startswith('Prevention:'):
                        current_key = 'prevention'
                        content['prevention'] = line.replace('Prevention:', '<b>Prevention:</b>')
                    elif current_key and line:  # Continue previous section if it's a continuation line
                        content[current_key] += ' ' + line

                # Create formatted title with the same indentation as content
                vuln_name = content['name'] if content['name'] else 'Unknown Vulnerability'
                title_style = ParagraphStyle(
                    'VulnerabilityTitle',
                    parent=styles['VulnerabilityTitle'],
                    leftIndent=20  # Match the CustomNormal style's left indent
                )
                formatted_title = f"<b>Vulnerability {vulnerability_count}: {vuln_name}</b>"
                story.append(Paragraph(formatted_title, title_style))
                story.append(Spacer(1, 6))

                # Add the rest of the content
                for key in ['description', 'impact', 'fix', 'prevention']:
                    if content[key]:
                        story.append(Paragraph(content[key], styles['CustomNormal']))
                        story.append(Spacer(1, 4))

                vulnerability_count += 1

            return story

        except Exception as e:
            logging.error(f"Error formatting repair section: {str(e)}")
            story.append(Paragraph("Error formatting repair suggestions", styles['CustomNormal']))
            return story


    def _create_severity_box(self, severity: str, count: int, color: str) -> Table:
        """Create a colored severity box"""
        styles = self._create_custom_styles()
        
        # 创建单元格内容
        data = [[
            Paragraph(f'<b>{severity}</b>', styles['CustomNormal']),
            Paragraph(f'<b>{count}</b>', styles['CustomNormal'])
        ]]
        
        # 设置表格样式
        t = Table(data, colWidths=[100, 50])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#FF0000')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#FFFFFF')),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        return t
    
    def extract_contract_info(self, contract_code: Union[str, List[str]], context: Dict[str, Any]) -> Dict[str, str]:
        """Extract contract information from function snippets"""
        try:
            if isinstance(contract_code, list):
                contract_code = "\n".join(str(line) for line in contract_code if line is not None)
            elif not isinstance(contract_code, str):
                contract_code = str(contract_code)

            code_lines = [line.strip() for line in contract_code.split('\n') if line.strip()]
            full_code = '\n'.join(code_lines)

            import re
            function_name = None
            function_pattern = re.compile(r'function\s+(\w+)\s*\([^)]*\)')
            func_match = function_pattern.search(full_code)
            if func_match:
                function_name = func_match.group(1)
            else:
                function_name = "Unknown Function"

            function_analysis = self.llm.get_response(full_code)
            if not function_analysis:
                function_analysis = "Function analysis unavailable"

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            return {
                "name": function_name,
                "function_type": function_analysis,
                "timestamp": timestamp
            }

        except Exception as e:
            logging.error(f"Error extracting contract info: {str(e)}")
            return {
                "name": "Unknown Function",
                "function_type": "Error analyzing function",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

   
    def _generate_executive_summary(self, context: Dict[str, Any]) -> str:
        """Generate executive summary based on vulnerability assessment"""
        try:
            risk_assessment = context.get("risk_assessment", {})
            if isinstance(risk_assessment, str):
                return "Unable to generate executive summary due to invalid risk assessment data."

            vulnerability_check = context.get("vulnerability_check", "").lower()
            risk_counts = risk_assessment.get("risk_counts", {})
            
            if not isinstance(risk_counts, dict):
                risk_counts = {}

            # Get vulnerability counts
            critical_count = risk_counts.get("Critical", 0)
            high_count = risk_counts.get("High", 0)
            medium_count = risk_counts.get("Medium", 0)
            low_count = risk_counts.get("Low", 0)
            
            if vulnerability_check == "vulnerable":
                total_vulns = sum(risk_counts.values())
                if total_vulns > 0:
                    return (
                        f"After auditing, we discovered {critical_count} Critical, {high_count} High, "
                        f"{medium_count} Medium and {low_count} Low severity vulnerabilities in the "
                        f"contract. These vulnerabilities could pose significant risks to the security "
                        f"and functionality of the contract. The contract owner and developers should "
                        f"address these issues promptly to prevent potential security threats."
                    )
                else:
                    return (
                        "After auditing, no specific vulnerability severity counts were available, but "
                        "potential security risks were identified. The contract owner and developers "
                        "should review and address any identified issues to ensure contract security."
                    )
            else:
                return (
                    "After auditing, we confirm that no critical or high-severity vulnerabilities "
                    "were found in the contract. The contract, as currently implemented, does not "
                    "present any obvious security risks. We recommend regular security audits and "
                    "code updates to maintain long-term security."
                )

        except Exception as e:
            logging.error(f"Error in _generate_executive_summary: {str(e)}")
            return "An error occurred while generating the executive summary."
    
    def _get_severity_colors(self) -> Dict[str, Any]:
        """Get color definitions for different severity levels"""
        return {
            'Critical': HexColor('#FFE6E6'),  # 浅红色
            'High': HexColor('#FFF3E0'),      # 浅橙色
            'Medium': HexColor('#FFF9C4'),    # 浅黄色
            'Low': HexColor('#E3F2FD')        # 浅蓝色
        }

    def _generate_findings_section(self, context: Dict[str, Any]) -> Dict:
        """Generate findings section with vulnerability information"""
        try:
            vulnerability_check = context.get("vulnerability_check", "").lower()
            risk_assessment = context.get("risk_assessment", {})
            
            if isinstance(risk_assessment, str):
                risk_assessment = {"risk_counts": {}, "predicted_details": [], "overall_assessment": ""}
            
            vuln_counts = risk_assessment.get("risk_counts", {})
            if isinstance(vuln_counts, str):
                vuln_counts = {}
                
            # Get total vulnerabilities from risk counts
            total_vulns = sum(vuln_counts.values())
            
            # 4.1 Vulnerability Statistics
            vulnerability_stats = {
                "Detection Result": (
                    "<b>Found Vulnerabilities</b>" if vulnerability_check == "vulnerable" 
                    else "<b>Not Found Vulnerabilities</b>"
                ),
                "Vulnerability Count": f"<b>Total of {total_vulns} vulnerabilities found</b>"
            }
            
            # 4.2 Vulnerability Severity Distribution
            severity_distribution = {
                "Critical": vuln_counts.get("Critical", 0),
                "High": vuln_counts.get("High", 0),
                "Medium": vuln_counts.get("Medium", 0),
                "Low": vuln_counts.get("Low", 0)
            }
            
            # 使用predicted_details生成漏洞详情
            vulnerability_details = []
            predicted_details = risk_assessment.get("predicted_details", [])
            
            for vuln in predicted_details:
                if isinstance(vuln, dict):
                    vulnerability_details.append(
                        f"- {vuln.get('vulnerability', 'Unknown')} ({vuln.get('level', 'Unknown')})\n"
                        f"  Description: {vuln.get('description', 'No description available')}\n"
                        f"  Impact: {vuln.get('impact', 'Impact not specified')}"
                    )
            
            vuln_details_text = "\n\n".join(vulnerability_details) if vulnerability_details else \
                "No significant vulnerabilities were identified in the current analysis."
            
            return {
                "4.1 Vulnerability Statistics": vulnerability_stats,
                "4.2 Vulnerability Severity Distribution": {
                    "Distribution": severity_distribution,
                    "Severity Colors": self._get_severity_colors(),
                    "Reference Table": self._get_reference_table()
                },
                "4.3 Vulnerability Details List": vuln_details_text
            }
                    
        except Exception as e:
            logging.error(f"Error in _generate_findings_section: {str(e)}")
            return {
                "4.1 Vulnerability Statistics": {
                    "Detection Result": "<b>Error</b>",
                    "Vulnerability Count": "<b>Error processing vulnerability count</b>"
                },
                "4.2 Vulnerability Severity Distribution": {
                    "Distribution": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
                    "Severity Colors": self._get_severity_colors(),
                    "Reference Table": self._get_reference_table()
                },
                "4.3 Vulnerability Details List": "Error processing vulnerability details"
            }


    def _get_reference_table(self) -> list:
        """Get vulnerability reference table"""
        return [
            {
                "Vulnerability Name": "Reentrancy (RE)",
                "Severity": "Critical",
                "Impact Scope": "May lead to theft of funds, contract failure, or complete control of contract permissions."
            },
            {
            "Vulnerability Name": "Access Control Missing (AC)",
            "Severity": "Critical",
            "Impact Scope": "Contract permissions may be maliciously controlled, potentially leading to owner replacement and fund theft."
            },
            {
                "Vulnerability Name": "Unchecked Low-level Call (ULC)",
                "Severity": "Critical",
                "Impact Scope": "Low-level calls cannot catch exceptions, potentially leading to failed contract calls or misoperations."
            },
            {
                "Vulnerability Name": "Integer Overflow/Underflow (IOU)",
                "Severity": "High",
                "Impact Scope": "May result in fund loss, severely impact core contract functionality, and cause calculation errors."
            },
            {
                "Vulnerability Name": "Denial of Service - DoS (DoS)",
                "Severity": "High",
                "Impact Scope": "Attackers can prevent normal contract operation through gas consumption or other resource exhaustion."
            },
            {
                "Vulnerability Name": "Flash Loan Vulnerability (FLV)",
                "Severity": "High",
                "Impact Scope": "Malicious users can manipulate market prices or contract states through flash loans, potentially leading to fund loss."
            },
            {
                "Vulnerability Name": "Front Running (FR)",
                "Severity": "High",
                "Impact Scope": "Attackers can manipulate transaction order to execute certain transactions first, leading to profit loss."
            },
            {
                "Vulnerability Name": "Timestamp Dependence (TD)",
                "Severity": "Medium",
                "Impact Scope": "Contract behavior depends on block timestamps which can be manipulated by attackers."
            },
            {
                "Vulnerability Name": "Block Info Dependence (BI)",
                "Severity": "Medium",
                "Impact Scope": "Contract relies on block information that can be manipulated by miners or predicted by attackers."
            },
            {
                "Vulnerability Name": "DoS with Gas Limit (DosGL)",
                "Severity": "Medium",
                "Impact Scope": "Gas limits during execution may cause contract suspension and prevent normal operation."
            },
            {
                "Vulnerability Name": "Unsafe Type Casting (UR)",
                "Severity": "Medium",
                "Impact Scope": "Type casting errors can lead to arithmetic overflow and contract logic errors."
            },
            {
                "Vulnerability Name": "Transaction Order Dependence (TOD)",
                "Severity": "Medium",
                "Impact Scope": "Attackers can manipulate transaction order affecting contract execution logic."
            },
            {
                "Vulnerability Name": "Outdated Compiler Version (OCV)",
                "Severity": "Low",
                "Impact Scope": "Using outdated compiler versions may expose contract to known vulnerabilities and incompatibilities."
            },
            {
                "Vulnerability Name": "Naming Convention (NC)",
                "Severity": "Low",
                "Impact Scope": "Non-standard naming conventions may lead to poor code readability and maintenance difficulties."
            },
            {
                "Vulnerability Name": "Redundant Code (RC)",
                "Severity": "Low",
                "Impact Scope": "Redundant code may increase gas costs and make contract more complex to maintain."
            }
    ]


    def _generate_detailed_analysis(self, context: Dict[str, Any]) -> list:
        """Generate detailed analysis section"""
        try:
            contract_info = context.get("contract_info", {})
            contract_name = contract_info.get("name", "Unknown Contract")
            code = context.get("contract_code", "Code not available")
            repair_suggestion = context.get("repair_suggestion", {})
            fixed_code = context.get("fixed_code", "")
            vulnerability_check = context.get("vulnerability_check", "").lower()
            
            if not isinstance(repair_suggestion, str):
                repair_suggestion = repair_suggestion.get("generated_suggestion", "")

            # 处理代码格式
            if isinstance(code, list):
                code = "\n".join(str(line) for line in code if line is not None)
            if isinstance(fixed_code, list):
                fixed_code = "\n".join(str(line) for line in fixed_code if line is not None)
            
            # 如果不是易受攻击的合约，返回标准信息
            if vulnerability_check != "vulnerable":
                return [{
                    "contract_name": contract_name,
                    "code_snippet": str(code).replace("<", "&lt;").replace(">", "&gt;"),
                    "repair_suggestion": "No vulnerabilities were detected; therefore, no remediation actions are required.",
                    "fixed_code": ""
                }]
            
            return [{
                "contract_name": contract_name,
                "code_snippet": str(code).replace("<", "&lt;").replace(">", "&gt;"),
                "repair_suggestion": repair_suggestion,
                "fixed_code": str(fixed_code).replace("<", "&lt;").replace(">", "&gt;")
            }]
                
        except Exception as e:
            logging.error(f"Error in _generate_detailed_analysis: {str(e)}")
            return [{
                "contract_name": "Error in Analysis",
                "code_snippet": "Error generating analysis",
                "repair_suggestion": f"Error: {str(e)}",
                "fixed_code": "Unable to generate fixed code due to error"
            }]


    def _generate_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate security recommendations"""
        return {
            "Content": "Based on the audit results, we recommend:",
            "Recommendations": [
                "Regularly conduct security audits to identify vulnerabilities and ensure safety.",
                "Continuously update the code based on industry best practices for long-term security.", 
                "Use the proxy pattern for upgradability and continuous monitoring.",
                "Implement error handling and fail-safes to manage failures safely.",
                "Minimize external calls and validate inputs to prevent reentrancy attacks."
            ]
        }

    
            
    def _generate_pdf(self, report_data: Dict[str, Any], output_path: str):
        """Generate PDF with custom styling and improved handling of long content"""
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=50, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = self._create_custom_styles()
        story = []

        # Add Title and Subtitle
        story.append(Paragraph("VulGuardPro", styles['ReportTitle']))
        story.append(Paragraph("Smart Contract Audit Report", styles['ReportSubtitle']))
        story.append(Spacer(1, 2))

        # 1. Contract Information
        story.append(Paragraph("1. Contract Information", styles['CustomHeading2']))
        contract_info = report_data.get("1. Contract Information", {})
        for key, value in contract_info.items():
            story.append(Paragraph(f"{key}: {value}", styles['CustomNormal']))
        story.append(Spacer(1, 20))

        # 2. Executive Summary
        story.append(Paragraph("2. Executive Summary", styles['CustomHeading2']))
        executive_summary = report_data.get("2. Executive Summary", {}).get("Content", "")
        story.append(Paragraph(executive_summary, styles['CustomNormal']))
        story.append(Spacer(1, 10))

        # 3. Methodology
        story.append(Paragraph("3. Methodology", styles['CustomHeading2']))
        methodology = report_data.get("3. Methodology", {}).get("Content", "")
        story.append(Paragraph(methodology, styles['CustomNormal']))
        story.append(Spacer(1, 10))

        # 4. Findings
        story.append(Paragraph("4. Findings", styles['MainTitle']))
        findings = report_data.get("4. Findings", {})

        # 4.1 Vulnerability Statistics
        story.append(Paragraph("4.1 Vulnerability Statistics", styles['CustomHeading2']))
        stats = findings.get("4.1 Vulnerability Statistics", {})
        for key, value in stats.items():
            story.append(Paragraph(f"{key}: {value}", styles['CustomNormal']))
        story.append(Spacer(1, 20))

        # 4.2 Vulnerability Severity Distribution
        story.append(Paragraph("4.2 Vulnerability Severity Distribution", styles['CustomHeading2']))
        distribution = findings.get("4.2 Vulnerability Severity Distribution", {})

        # Create severity distribution table
        severity_data = [["Critical", "High", "Medium", "Low"]]
        severity_counts = [str(distribution.get("Distribution", {}).get(sev, 0)) 
                            for sev in ["Critical", "High", "Medium", "Low"]]
        severity_data.append(severity_counts)

        severity_table = Table(severity_data, colWidths=[125] * 4)
        severity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#FFE6E6')),  # Critical
            ('BACKGROUND', (1, 0), (1, -1), HexColor('#FFF3E0')),  # High
            ('BACKGROUND', (2, 0), (2, -1), HexColor('#FFF9C4')),  # Medium
            ('BACKGROUND', (3, 0), (3, -1), HexColor('#E3F2FD')),  # Low
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.red),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('PADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(severity_table)
        story.append(Spacer(1, 20))

        # 4.3 Vulnerability Reference Table
        story.append(Paragraph("4.3 Vulnerability Reference Table", styles['CustomHeading2']))
        reference_data = [[
            Paragraph("<b>Vulnerability Name</b>", styles['TableHeader']),
            Paragraph("<b>Severity</b>", styles['TableHeader']),
            Paragraph("<b>Impact Scope</b>", styles['TableHeader'])
        ]]

        for item in distribution.get("Reference Table", []):
            reference_data.append([
                Paragraph(item["Vulnerability Name"], styles['CustomNormal']),
                Paragraph(item["Severity"], styles['CustomNormal']),
                Paragraph(item["Impact Scope"], styles['CustomNormal'])
            ])

        reference_table = Table(reference_data, colWidths=[180, 100, 220])
        reference_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4A90E2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),                  # 顶部内边距
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#DDDDDD')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(reference_table)
        story.append(Spacer(1, 20))

        # 5. Detailed Analysis
        story.append(Paragraph("5. Detailed Analysis", styles['MainTitle']))
        detailed_analysis = report_data.get("5. Detailed Analysis", [])

        for analysis in detailed_analysis:
            # 5.1 Contract Name
            story.append(Paragraph("5.1 Contract Name:", styles['SectionTitle']))
            story.append(Paragraph(analysis.get('contract_name', 'N/A'), styles['CustomNormal']))
            story.append(Spacer(1, 10))

            # 5.2 Source Code
            story.append(Paragraph("5.2 Source Code:", styles['SectionTitle']))
            code_snippet = analysis.get("code_snippet", "")
            if code_snippet:
                story.append(Paragraph(code_snippet, styles['CodeBlock']))
            story.append(Spacer(1, 10))

            # 5.3 Repair Suggestion
            story.append(Paragraph("5.3 Repair Suggestion:", styles['SectionTitle']))
            repair_suggestion = analysis.get("repair_suggestion", "")

            # 使用 _format_repair_section 方法
            story.extend(self._format_repair_section(repair_suggestion, styles))

            # 5.4 Fixed Code
            story.append(Paragraph("5.4 Fixed Code:", styles['SectionTitle']))
            fixed_code = analysis.get("fixed_code", "")
            if fixed_code:
                story.append(Paragraph(fixed_code, styles['CodeBlock']))
            else:
                # Use CodeBlock style even if no code is present
                story.append(Paragraph("No vulnerabilities were detected, therefore no repair actions are required.", styles['CodeBlock']))
            story.append(Spacer(1, 10))

        # 6. Summary and Recommendations
        story.append(Paragraph("6. Summary and Recommendations", styles['CustomHeading2']))
        recommendations = report_data.get("6. Summary and Recommendations", {})
        story.append(Paragraph(recommendations.get("Content", ""), styles['CustomNormal']))
        for rec in recommendations.get("Recommendations", []):
            story.append(Paragraph(f"\u2022 {rec}", styles['CustomNormal']))
        story.append(Spacer(1, 20))

        # 7. Disclaimer
        story.append(Paragraph("7. Disclaimer", styles['CustomHeading2']))
        disclaimer = report_data.get("7. Disclaimer", {}).get("Content", "")
        story.append(Paragraph(disclaimer, styles['CustomNormal']))

        # Build the document
        try:
            doc.build(story)
        except Exception as e:
            logging.error(f"Error building PDF: {str(e)}")
            raise


    def analyze(self, contract_code: Union[str, List[str]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate complete audit report with comprehensive error handling"""
        if context is None:
            context = {}
            
        try:
            # 规范化contract_code
            if isinstance(contract_code, list):
                contract_code = "\n".join(str(line) for line in contract_code if line is not None)
            elif not isinstance(contract_code, str):
                contract_code = str(contract_code)
                
            # 正确提取漏洞检测结果
            vulnerability_check = context.get("vulnerability_check", "")
            if not vulnerability_check and "vulnerability_check" in context.get("vulnerability_results", {}):
                vulnerability_check = context["vulnerability_results"]["vulnerability_check"]
                
            # 更新context
            context["contract_code"] = contract_code
            context["vulnerability_check"] = vulnerability_check
            
            # 获取合约信息
            contract_info = self.extract_contract_info(contract_code, context)
            if not isinstance(contract_info, dict):
                contract_info = {
                    "name": "Unknown Contract",
                    "function_type": "Unknown Function Type",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
            # 将合约信息添加到context
            context["contract_info"] = contract_info
            
            # 确保风险评估结果可用
            risk_assessment = context.get("risk_assessment", {})
            if isinstance(risk_assessment, str):
                risk_assessment = {}
            if "risk_assessment" in context.get("risk_results", {}):
                risk_assessment = context["risk_results"]["risk_assessment"]
            context["risk_assessment"] = risk_assessment

            # 生成报告数据
            report_data = {
                "1. Contract Information": {
                    "Analyzed Object": contract_info.get("name", "Unknown"),
                    "Contract Function": contract_info.get("function_type", "Unknown"),
                    "Detection Result": str(vulnerability_check),
                    "Audit Time": contract_info.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                },
                "2. Executive Summary": {
                    "Content": self._generate_executive_summary(context)
                },
                "3. Methodology": {
                    "Content": ("Our audit process included static analysis, dynamic testing, and manual code review. "
                            "We used advanced vulnerability detection models and compared against known vulnerability patterns.")
                },
                "4. Findings": self._generate_findings_section(context),
                "5. Detailed Analysis": self._generate_detailed_analysis(context),
                "6. Summary and Recommendations": self._generate_recommendations(context),
                "7. Disclaimer": {
                    "Content": ("This audit report represents our best effort in identifying potential security vulnerabilities. "
                            "However, we cannot guarantee that all possible vulnerabilities have been identified.")
                }
            }

            # 创建输出目录
            output_dir = Path("output/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成PDF文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = output_dir / f"auditreport_{timestamp}.pdf"
            pdf_path_str = str(pdf_path.absolute())
            
            # 生成PDF
            self._generate_pdf(report_data, pdf_path_str)
            
            return {
                "report": {
                    "data": report_data,
                    "pdf_path": str(pdf_path)
                }
            }

        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logging.error(error_msg)
            logging.error(f"Context: {context}")
            
            return {
                "report": {
                    "error": error_msg,
                    "data": {
                        "1. Contract Information": {
                            "Error": "Failed to generate report"
                        },
                        "2. Executive Summary": {
                            "Content": error_msg
                        }
                    },
                    "pdf_path": None
                }
            }


