import re

with open("src/array/ops.rs", "r", encoding="utf-8") as f:
    text = f.read()

# We replace occurrences of `let X = self.data();` with:
# `let contig = self.to_contiguous(); let X = contig.data();`
# Wait, if `other.data()` is used: `let contig_other = other.to_contiguous(); let Y = contig_other.data();`

def replacer_self(m):
    return f"let contig_self = self.to_contiguous();\n        let {m.group(1)} = contig_self.data();"

def replacer_other(m):
    return f"let contig_other = other.to_contiguous();\n        let {m.group(1)} = contig_other.data();"

def replacer_v(m):
    return f"let contig_self = self.to_contiguous();\n        let {m.group(1)} = contig_self.data();"

# Carefully map specific known methods that trigger panics on transposed:
text = re.sub(r'pub fn sum_all\(&self\) -> f64 \{\s*let s = self\.data\(\);', 
              r'pub fn sum_all(&self) -> f64 {\n        let contig = self.to_contiguous();\n        let s = contig.data();', text)

text = re.sub(r'pub fn prod\(&self\)    -> f64 \{ \s*let d = self\.data\(\);', 
              r'pub fn prod(&self)    -> f64 { \n        let contig = self.to_contiguous();\n        let d = contig.data();', text)

text = re.sub(r'pub fn variance\(&self\) -> f64 \{\s*let n = self\.len\(\);\s*if n < 2 \{ return f64::NAN; \}\s*let mu = self\.mean\(\);\s*let sd = self\.data\(\);', 
              r'pub fn variance(&self) -> f64 {\n        let n = self.len();\n        if n < 2 { return f64::NAN; }\n        let mu = self.mean();\n        let contig = self.to_contiguous();\n        let sd = contig.data();', text)

for func_name in ["min", "max", "argmin", "argmax"]:
    text = re.sub(rf'pub fn {func_name}\(&self\) -> (f64|usize) \{\s*let d = self\.data\(\);', 
                  rf'pub fn {func_name}(&self) -> \1 {{\n        let contig = self.to_contiguous();\n        let d = contig.data();', text)


# The ones under axis=None for sum:
text = re.sub(r'Some\(0\) => \{\s*let \(r, c\) = \(self\.nrows\(\), self\.ncols\(\)\);\s*let d = self\.data\(\);',
              r'Some(0) => {\n                let (r, c) = (self.nrows(), self.ncols());\n                let contig = self.to_contiguous();\n                let d = contig.data();', text)
text = re.sub(r'Some\(1\) => \{\s*let \(r, c\) = \(self\.nrows\(\), self\.ncols\(\)\);\s*let d = self\.data\(\);',
              r'Some(1) => {\n                let (r, c) = (self.nrows(), self.ncols());\n                let contig = self.to_contiguous();\n                let d = contig.data();', text)

# mean_axis0, mean_axis1, std_axis0, var_axis0
for func_name in ["mean_axis0", "mean_axis1", "std_axis0", "var_axis0"]:
    text = re.sub(rf'pub fn {func_name}\(&self\) -> PyResult<Vector> \{([^}}]*?)let d = self\.data\(\);',
                  rf'pub fn {func_name}(&self) -> PyResult<Vector> {{\1let contig = self.to_contiguous();\n        let d = contig.data();', text, flags=re.DOTALL)

# comparisons
for comp in ["gt", "lt", "ge", "le", "eq_scalar", "ne_scalar", "isnan", "isfinite", "isinf"]:
    text = re.sub(rf'pub fn {comp}\(&self(, val: f64)?\)  ?-> Vec<bool> {{ self\.data\(\)',
                  rf'pub fn {comp}(&self\1) -> Vec<bool> {{ self.to_contiguous().data()', text)

text = re.sub(r'pub fn where_scalar\(&self, mask: Vec<bool>, other: f64\) -> PyResult<Self> \{([^\}]+?)let data: Vec<f64> = self\.data\(\)',
              r'pub fn where_scalar(&self, mask: Vec<bool>, other: f64) -> PyResult<Self> {\1let contig = self.to_contiguous();\n        let data: Vec<f64> = contig.data()', text, flags=re.DOTALL)


with open("src/array/ops.rs", "w", encoding="utf-8") as f:
    f.write(text)
