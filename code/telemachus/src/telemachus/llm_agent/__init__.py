"""TELEMACHUS for language agents: when to answer, think, use a tool, or ask.

A small local LLM solves grade-school math (GSM8K). For each question it can
escalate through increasingly expensive information sources:

  ANSWER  direct answer                       (few tokens)
  THINK   chain-of-thought                    (hundreds of tokens)
  TOOL    program-of-thought, executed safely (tokens + sandboxed run)
  VOTE    self-consistency over k samples     (k x THINK)
  EXPERT  human answer                        (fixed, large cost)

A gate decides after each stage whether to stop. Learn-then-Test (shared
with the robot domain, telemachus.calibration) certifies that the fraction
of wrong final answers is <= alpha with probability >= 1 - delta.
"""
