class Extractor:
    def __init__(self):
        pass

    def __call__(self, msg):
        try:
            if "```python" in msg:
                return msg.split("```python")[1].split("```")[0]
            elif "```plaintext" in msg:
                return msg.split("```plaintext")[1].split("```")[0]
            elif "```text" in msg:
                return msg.split("```text")[1].split("```")[0]
            elif "```" in msg:
                return msg.split("```")[1].split("```")[0]
            else:
                return None
        except:
            return None
                
    def extract(self, msg):
        if "```" in msg:
            return msg.split("```")[1].split("```")[0]
        else:
            return None

    def extract_text(self, msg):
        if "```text" in msg:
            return msg.split("```text")[1].split("```")[0]
        else:
            return self.extract(msg)

    def extract_plain_text(self, msg):
        if "```plaintext" in msg:
            return msg.split("```plaintext")[1].split("```")[0]
        text = self.extract_text(msg)
        if text:
            return text
        # Fallback: handle unfenced five-element blocks with Markdown headings.
        headings = ["## Sets", "## Parameters", "## Variables", "## Objective", "## Constraints"]
        first_idx = None
        for h in headings:
            idx = msg.find(h)
            if idx != -1:
                first_idx = idx
                break
        if first_idx is None:
            return None
        tail = msg[first_idx:]
        # Ensure it looks like a five-element block by finding at least 3 headings.
        found = sum(1 for h in headings if h in tail)
        if found < 3:
            return None
        return tail

    def extract_python(self, msg):
        if "```python" in msg:
            return msg.split("```python")[1].split("```")[0]
        else:
            return self.extract(msg)
