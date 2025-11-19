def interpret_p_value(p):
    if p < 0.001:
        return "Highly significant results (p < 0.001). Strong evidence against null hypothesis."
    elif p < 0.05:
        return "Statistically significant results (p < 0.05)."
    else:
        return "No significant difference (p > 0.05). Null hypothesis is not rejected."

def generic_interpretation(test_name, p):
    text = f"""
## 📌 Interpretation for {test_name}
- P-value: **{round(p, 4)}**
- Conclusion: {interpret_p_value(p)}
"""
    return text
def interpret_anova_table(anova_table):
    # Extract F and p for main effect (first row)
    if "PR(>F)" in anova_table.columns:
        p = anova_table["PR(>F)"].iloc[0]
        return generic_interpretation("ANOVA", p)
    return "Could not extract p-value."
