from typing import Dict, Any, List 
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskCriteria:
    """Risk level evaluation criteria"""
    Critical: List[str] = field(default_factory=lambda: [
        "Reentrancy",
        "Access control issues", 
        "Unchecked external calls"
    ])
    
    High: List[str] = field(default_factory=lambda: [
        "Integer overflow",
        "Integer underflow", 
        "Denial of Service - DoS",
        "Flash loan vulnerabilities",
        "Front-running vulnerabilities"
    ])
    
    Medium: List[str] = field(default_factory=lambda: [
        "Timestamp dependence",
        "Block information dependence",
        "Gas limit issues", 
        "Transaction Ordering Dependency",
        "Unsafe Type Casting"
    ])
    
    Low: List[str] = field(default_factory=lambda: [
        "Outdated compiler version",
        "Naming conventions",
        "Redundant code"
    ])

class RiskAssessor:
    def __init__(self):
        self.risk_criteria = RiskCriteria()
        self.risk_weights = {
            "Critical": 1.0,
            "High": 0.8,
            "Medium": 0.5, 
            "Low": 0.2
        }

    def _parse_repair_suggestions(self, repair_suggestion: str) -> List[Dict]:
        """Parse repair suggestions to extract vulnerabilities"""
        vulnerabilities = []
        
        # Split by "Vulnerability X:" pattern
        sections = re.split(r'Vulnerability\s+\d+:', repair_suggestion)
        
        for section in sections[1:]:  # Skip the first empty section
            try:
                # Extract fields using format patterns
                name_match = re.search(r'Name:\s*(.*?)(?=Description:|$)', section, re.DOTALL)
                desc_match = re.search(r'Description:\s*(.*?)(?=Impact:|$)', section, re.DOTALL)
                impact_match = re.search(r'Impact:\s*(.*?)(?=Fix:|$)', section, re.DOTALL)
                
                if name_match and desc_match and impact_match:
                    vuln = {
                        'name': name_match.group(1).strip(),
                        'description': desc_match.group(1).strip(),
                        'impact': impact_match.group(1).strip()
                    }
                    vulnerabilities.append(vuln)
                    
            except Exception as e:
                logger.error(f"Error parsing vulnerability section: {str(e)}")
                continue
                
        return vulnerabilities

    def _determine_risk_level(self, vulnerability: Dict) -> str:
        """Determine risk level for a single vulnerability"""
        vuln_info = f"{vulnerability.get('name', '')} {vulnerability.get('description', '')} {vulnerability.get('impact', '')}".lower()
        
        # Check against each risk level's patterns
        for level in ["Critical", "High", "Medium", "Low"]:
            patterns = getattr(self.risk_criteria, level)
            if any(pattern.lower() in vuln_info for pattern in patterns):
                return level
                
        # Check impact-based indicators
        impact_indicators = {
            "Critical": ["fund loss", "complete control", "arbitrary execution", "security breach"],
            "High": ["denial of service", "unauthorized access", "manipulation"],
            "Medium": ["state inconsistency", "gas inefficiency", "potential impact"],
            "Low": ["best practice", "code style", "optimization"]
        }
        
        for level, indicators in impact_indicators.items():
            if any(indicator in vuln_info for indicator in indicators):
                return level
        
        return "Medium"  # Default risk level

    def analyze(self, contract_code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contract risk level"""
        try:
            # Check vulnerability status
            vulnerability_check = context.get("vulnerability_check", "Safe")
            repair_suggestion = context.get("repair_suggestion", "")
            
            if vulnerability_check.lower() == "safe":
                return {
                    "risk_assessment": {
                        "highest_risk": "Safe",
                        "predicted_details": [],
                        "risk_counts": dict.fromkeys(self.risk_weights.keys(), 0),
                        "risk_score": 0.0,
                        "overall_assessment": "No security vulnerabilities were detected in the contract."
                    }
                }

            # Parse vulnerabilities from repair suggestions
            vulnerabilities = self._parse_repair_suggestions(repair_suggestion)
            
            # Initialize risk counts
            risk_counts = {level: 0 for level in self.risk_weights.keys()}
            predicted_details = []
            
            # Analyze each vulnerability
            for vuln in vulnerabilities:
                risk_level = self._determine_risk_level(vuln)
                risk_counts[risk_level] += 1
                
                predicted_details.append({
                    "vulnerability": vuln.get('name', 'Unknown'),
                    "level": risk_level,
                    "description": vuln.get('description', ''),
                    "impact": vuln.get('impact', '')
                })

            # Determine highest risk level
            highest_risk = "Safe"
            for level in ["Critical", "High", "Medium", "Low"]:
                if risk_counts.get(level, 0) > 0:
                    highest_risk = level
                    break

            # Calculate risk score
            total_score = 0.0
            for level, count in risk_counts.items():
                if level in self.risk_weights:
                    total_score += count * self.risk_weights[level]
            max_score = sum(self.risk_weights.values())
            risk_score = min(1.0, total_score / max_score) if max_score > 0 else 0.0

            # Generate overall assessment
            overall_assessment = self._generate_assessment(highest_risk, risk_counts, risk_score)

            return {
                "risk_assessment": {
                    "highest_risk": highest_risk,
                    "predicted_details": predicted_details,
                    "risk_counts": risk_counts,
                    "risk_score": risk_score,
                    "overall_assessment": overall_assessment
                }
            }

        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {
                "risk_assessment": {
                    "highest_risk": "Unknown",
                    "predicted_details": [],
                    "risk_counts": {},
                    "risk_score": 0.0,
                    "overall_assessment": f"Error during risk assessment: {str(e)}"
                }
            }

    def _generate_assessment(self, highest_risk: str, risk_counts: Dict[str, int], risk_score: float) -> str:
        """Generate overall risk assessment summary"""
        total_vulns = sum(risk_counts.values())
        
        assessment = f"Security analysis identified {total_vulns} vulnerabilities with highest risk level {highest_risk}. "
        
        # Add risk distribution
        risk_details = []
        for level in ["Critical", "High", "Medium", "Low"]:
            count = risk_counts.get(level, 0)
            if count > 0:
                risk_details.append(f"{count} {level.lower()}")
        
        if risk_details:
            assessment += f"Risk distribution: {', '.join(risk_details)}. "
            
        assessment += f"Overall risk score: {risk_score:.2%}. "
        
        # Add recommendation based on risk level
        recommendations = {
            "Critical": "IMMEDIATE ACTION REQUIRED: Fix all critical vulnerabilities before deployment.",
            "High": "URGENT: Prioritize fixing high-risk vulnerabilities and implement additional monitoring.",
            "Medium": "IMPORTANT: Plan to address vulnerabilities in the next update cycle.",
            "Low": "ADVISORY: Address low-risk issues during routine maintenance."
        }
        
        if highest_risk in recommendations:
            assessment += f"\n{recommendations[highest_risk]}"
            
        return assessment