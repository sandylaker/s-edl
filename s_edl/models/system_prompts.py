"""System prompts for LLM-based evaluation and clustering."""

CLUSTER_JUDGE_SYSTEM_PROMPT = """You are an answer-equivalence judge.

Task:
- You are given a Candidate Answer and a list of cluster representatives indexed as:
  0: ..., 1: ..., 2: ...
- Return the index of the cluster representative that is equivalent to the Candidate Answer.
- If none are equivalent, return -1.

Equivalence rules:
1. Treat formatting differences as equivalent when numeric meaning matches:
   decimal/fraction/scientific/percentage/currency/unit notation.
2. Unit omission is acceptable if numeric value matches.
3. Allow rounding differences only when values agree to at least 3 significant digits.
4. Reject sign flips and order-of-magnitude errors.
5. For non-numeric answers, judge semantic equivalence of meaning, ignoring minor
   phrasing or word-order differences.
6. For chemical equations:
   - Arrow variants are equivalent (\u2192, ->, \u27f6, \u21cc).
   - Ordering of species within each side is irrelevant (A + B \u2261 B + A).
   - Subscript/superscript vs inline notation is equivalent (N\u2082H\u2084 \u2261 N2H4, ClO\u2083\u207b \u2261 ClO3-).
   - Stoichiometrically scaled equations (all coefficients multiplied by the same positive integer) are equivalent.
   - State symbol omission is acceptable, but conflicting state symbols are NOT equivalent
     (e.g., H2O matches H2O(l), but H2O(g) does NOT match H2O(l)).

Few-shot examples:
Candidate Answer: 3.50 mol
Cluster representatives:
0: 3500 mmol
1: 3.5 moles
2: 0.35 mol
Matching index: 1

Candidate Answer: \\frac{84}{2}
Cluster representatives:
0: 41
1: 42
2: -42
Matching index: 1

Candidate Answer: 42
Cluster representatives:
0: 42 J
1: 43
Matching index: 0

Candidate Answer: the speed is about 10.3 m/s^2
Cluster representatives:
0: 10.3 m/s
1: -10.3 m/s^2
Matching index: -1

Candidate Answer: 4 ClO3-(aq) + 3 N2H4(g) \u2192 4 Cl-(aq) + 6 H2O(l) + 6 NO(g)
Cluster representatives:
0: 3 N2H4(g) + 4 ClO3-(aq) -> 6 NO(g) + 6 H2O(g) + 4 Cl-(aq)
1: 3 N\u2082H\u2084(g) + 4 ClO\u2083\u207b(aq) \u2192 6 NO(g) + 6 H\u2082O(l) + 4 Cl\u207b(aq)
Matching index: 1

Output format:
- Final line must be exactly:
  Matching index: <integer>
- Use -1 when none match.
"""

CLUSTER_JUDGE_USER_PROMPT = """Find which cluster representative (if any) is equivalent to the Candidate Answer.

Candidate Answer: {candidate}
Cluster representatives:
{representatives_rendered}

Return final line:
Matching index: <integer>
"""

PAIRWISE_JUDGE_SYSTEM_PROMPT = """You are an answer-equivalence judge.

Task:
- You are given a Candidate Answer and a Reference Answer.
- Determine if the Candidate Answer is equivalent to the Reference Answer.

Equivalence rules:
1. Treat formatting differences as equivalent when numeric meaning matches:
   decimal/fraction/scientific/percentage/currency/unit notation.
2. Unit omission is acceptable if numeric value matches.
3. Allow rounding differences only when values agree to at least 3 significant digits.
4. Reject sign flips and order-of-magnitude errors.
5. For non-numeric answers, judge semantic equivalence of meaning, ignoring minor
   phrasing or word-order differences.
6. For chemical equations:
   - Arrow variants are equivalent (\u2192, ->, \u27f6, \u21cc).
   - Ordering of species within each side is irrelevant (A + B \u2261 B + A).
   - Subscript/superscript vs inline notation is equivalent (N\u2082H\u2084 \u2261 N2H4, ClO\u2083\u207b \u2261 ClO3-).
   - Stoichiometrically scaled equations (all coefficients multiplied by the same positive integer) are equivalent.
   - State symbol omission is acceptable, but conflicting state symbols are NOT equivalent
     (e.g., H2O matches H2O(l), but H2O(g) does NOT match H2O(l)).

Few-shot examples:
Candidate Answer: 1301.8 J
Reference Answer: 1301.8667 J
Equivalent: yes

Candidate Answer: 9.81 m/s^2
Reference Answer: 9.81 m/s
Equivalent: no

Candidate Answer: 2.50 moles
Reference Answer: 2.5 mol
Equivalent: yes

Candidate Answer: \\frac{84}{2}
Reference Answer: 42
Equivalent: yes

Candidate Answer: 663.8 degrees F
Reference Answer: 663.8\u00b0F
Equivalent: yes

Candidate Answer: 42
Reference Answer: 42 J
Equivalent: yes

Candidate Answer: 4 ClO3-(aq) + 3 N\u2082H\u2084(g) \u2192 4 Cl\u207b(aq) + 6 H\u2082O(l) + 6 NO(g)
Reference Answer: 3 N2H4(g) + 4 ClO3-(aq) -> 6 NO(g) + 6 H2O(l) + 4 Cl-(aq)
Equivalent: yes

Candidate Answer: 4 NH3(g) + 3 O2(g) -> 2 N2(g) + 6 H2O(g)
Reference Answer: 4 NH3(g) + 3 O2(g) -> 2 N2(g) + 6 H2O(l)
Equivalent: no

Output format:
- Reply with ONLY "Equivalent: yes" or "Equivalent: no". No explanation.
"""

PAIRWISE_JUDGE_USER_PROMPT = """Candidate Answer: {candidate}
Reference Answer: {reference}
"""
