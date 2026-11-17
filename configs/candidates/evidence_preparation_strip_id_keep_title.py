# 变体 A：只去掉 [文档 id] 前缀，保留标题。
# 与 evidence_preparation_strip_metadata.py（id + 标题都删）构成最小对照：
# 两版只差“标题是否保留”，逐题配对差可判定负向效果是否来自标题丢失。
# 注意：不能写模块级 docstring —— validate_source 要求文件体只有 CandidateModule 一个类。


class CandidateModule:
    def run(self, inputs):
        parts = []
        for doc in inputs.get("documents", []):
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            if title and text:
                block = title + "\n" + text
            elif text:
                block = text
            else:
                block = title
            if block:
                parts.append(block)
        return {"context": "\n\n".join(parts)[:9000]}
