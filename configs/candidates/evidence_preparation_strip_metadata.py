class CandidateModule:
    def run(self, inputs):
        parts = []
        for doc in inputs.get("documents", []):
            text = str(doc.get("text", "")).strip()
            if text:
                parts.append(text)
        return {"context": "\n\n".join(parts)[:9000]}
