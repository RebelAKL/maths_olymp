import re
from typing import Dict, Any

class SyntaxChecker:
    MATH_KEYWORDS = ['let', 'assume', 'therefore', 'contradiction', 'qed']
    MATH_PATTERNS = [
        r'\$.*?\$',  # LaTeX math
        r'\\begin\{proof\}.*?\\end\{proof\}',  # Proof environment
        r'\bdef\b',  # Definitions
    ]
    
    def check(self, solution: str) -> Dict[str, Any]:
        results = {
            'has_proof_structure': False,
            'math_keywords': [],
            'latex_math': False,
            'errors': []
        }
        
        # Check proof structure
        if any(keyword in solution.lower() for keyword in self.MATH_KEYWORDS):
            results['has_proof_structure'] = True
            results['math_keywords'] = [
                kw for kw in self.MATH_KEYWORDS if kw in solution.lower()]
            
        # Check LaTeX math
        if re.search(self.MATH_PATTERNS[0], solution):
            results['latex_math'] = True
            
        # Validate overall structure
        if not results['has_proof_structure'] and not results['latex_math']:
            results['errors'].append('Missing mathematical structure')
            
        return results