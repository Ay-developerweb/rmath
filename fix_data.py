import re

with open("src/array/ops.rs", "r", encoding="utf-8") as f:
    content = f.read()

# Replace self.data() with self.to_contiguous().data() in areas where it isn't broadcast_op fast paths
# We will just replace all `self.data()` and `other.data()`.
# Wait, broadcast_op fast paths explicitly check `is_contiguous()` earlier, so to_contiguous() will be an O(1) no-op!
# It will just clone the struct and then data() will return the slice immediately. 
# So it's totally fine to replace ALL of them!

content = content.replace("self.data()", "self.to_contiguous().data()")
content = content.replace("other.data()", "other.to_contiguous().data()")

with open("src/array/ops.rs", "w", encoding="utf-8") as f:
    f.write(content)
