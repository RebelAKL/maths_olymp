class GeometryVerifier:
    def __init__(self):
        self.rules = {
            'triangle': ['sum_of_angles', 'pythagoras'],
            'circle': ['arc_length', 'sector_area'],
            'polygon': ['interior_angles']
        }
        
    def check(self, solution: str, diagram_constraints: dict) -> dict:
        violations = []
        used_rules = []
        
        # Check for rule applications
        for shape in self.rules:
            if shape in solution.lower():
                for rule in self.rules[shape]:
                    if rule in solution:
                        used_rules.append(rule)
                        if not self._validate_rule(rule, diagram_constraints):
                            violations.append(f"Misapplied {rule}")
                            
        return {
            'used_rules': used_rules,
            'violations': violations,
            'is_valid': len(violations) == 0
        }
        
    def _validate_rule(self, rule: str, constraints: dict) -> bool:
        # Implementation would use geometric constraint checking
        return True  # Simplified for example